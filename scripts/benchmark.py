"""Score transcription accuracy against a reference transcript.

Runs a recording through exactly the same segmenter (Silero VAD) and ASR
backend the live app uses -- just without real-time pacing, so an hour of audio
takes minutes -- then compares the result with a human transcript.

    python scripts/benchmark.py --audio bench/okkei72.wav \
        --reference bench/okkei72_ref.txt --language "" --translate 12

Reports, for Japanese/Chinese (which have no spaces, so CER is the standard
metric):

* raw CER            -- exact characters, after removing punctuation/space
* reading CER        -- both sides converted to hiragana first, so writing the
                        same word in kanji vs kana (e.g. 私 vs わたし) is not
                        counted as a recognition error
* error breakdown    -- substitutions / deletions / insertions, long deletion
                        runs (missed speech) and insertion runs (hallucination)
* speed              -- real-time factor and per-segment latency
* language detection -- how often auto-detect picked each language

Requires the model servers to be running, plus: pip install rapidfuzz pykakasi
The media and reference transcript are not part of the repo (copyright).
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtsubs.asr import create_asr  # noqa: E402
from rtsubs.audio import dsp  # noqa: E402
from rtsubs.audio.segmenter import Segmenter  # noqa: E402
from rtsubs.audio.vad import create_vad  # noqa: E402
from rtsubs.config import SAMPLE_RATE, AppConfig  # noqa: E402
from rtsubs.translate import TranslationRequest, create_translator  # noqa: E402


# ---------------------------------------------------------------- text
def normalize(text: str) -> str:
    """NFKC, lowercase, and drop punctuation, symbols, spaces and controls."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] not in ("P", "S", "Z", "C")
    )


_KAKASI = None


def to_reading(text: str) -> str:
    """Convert to hiragana so orthographic variants compare equal."""
    global _KAKASI
    if _KAKASI is None:
        import pykakasi

        _KAKASI = pykakasi.kakasi()
    return "".join(item["hira"] for item in _KAKASI.convert(text))


def error_rate(ref: str, hyp: str) -> dict:
    from rapidfuzz.distance import Levenshtein

    ops = Levenshtein.editops(ref, hyp)
    counts = collections.Counter(op.tag for op in ops)
    n = max(1, len(ref))
    return {
        "cer": len(ops) / n,
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
        "substitutions": counts.get("replace", 0),
        "deletions": counts.get("delete", 0),
        "insertions": counts.get("insert", 0),
        "_ops": ops,
    }


def runs(ops, tag: str, min_len: int) -> list[int]:
    """Lengths of consecutive same-type edits: long ones = missed/extra speech."""
    lengths, current, last = [], 0, None
    for op in ops:
        pos = op.src_pos if tag == "delete" else op.dest_pos
        if op.tag == tag and last is not None and pos == last + 1:
            current += 1
        else:
            if current >= min_len:
                lengths.append(current)
            current = 1 if op.tag == tag else 0
        last = pos if op.tag == tag else None
    if current >= min_len:
        lengths.append(current)
    return lengths


# --------------------------------------------------------------- audio
def segment(audio: np.ndarray, cfg: AppConfig):
    segmenter = Segmenter(create_vad(cfg.vad), cfg.vad, cfg.segmenter)
    finals = []
    for i in range(0, audio.size, 512):
        finals.extend(e for e in segmenter.feed(audio[i : i + 512]) if e.is_final)
    finals.extend(segmenter.flush())
    return finals


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--language", default="", help='spoken language, "" = auto-detect')
    ap.add_argument("--translate", type=int, default=0, help="translate N sample lines")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = AppConfig.load()
    cfg.asr.language = args.language
    asr = create_asr(cfg.asr)
    ok, detail = asr.health()
    if not ok:
        print(f"ASR server not ready: {detail}", file=sys.stderr)
        return 1

    audio = dsp.read_wav_mono(str(args.audio), SAMPLE_RATE)
    duration = audio.size / SAMPLE_RATE
    print(f"audio: {duration / 60:.1f} min | language: {args.language or 'auto-detect'}")

    t0 = time.perf_counter()
    segments = segment(audio, cfg)
    vad_time = time.perf_counter() - t0
    speech = sum(s.duration for s in segments)
    print(f"VAD: {len(segments)} segments, {speech / 60:.1f} min of speech ({vad_time:.0f}s)")

    results = []
    asr_start = time.perf_counter()
    for n, seg in enumerate(segments, 1):
        tr = asr.transcribe(seg.audio, language=args.language)
        results.append(
            {
                "start": round(seg.start_time, 2),
                "duration": round(seg.duration, 2),
                "text": tr.text,
                "language": tr.language,
                "latency": round(tr.latency, 3),
            }
        )
        if n % 50 == 0 or n == len(segments):
            elapsed = time.perf_counter() - asr_start
            eta = elapsed / n * (len(segments) - n)
            print(f"  {n}/{len(segments)} segments, ETA {eta:.0f}s", flush=True)
    asr_time = time.perf_counter() - asr_start

    hyp_text = "".join(r["text"] for r in results)
    ref_text = args.reference.read_text(encoding="utf-8")

    ref_n, hyp_n = normalize(ref_text), normalize(hyp_text)
    raw = error_rate(ref_n, hyp_n)
    reading = error_rate(normalize(to_reading(ref_n)), normalize(to_reading(hyp_n)))

    ops = raw.pop("_ops")
    reading.pop("_ops")
    subs = collections.Counter(
        (ref_n[o.src_pos], hyp_n[o.dest_pos]) for o in ops if o.tag == "replace"
    )
    dels = collections.Counter(ref_n[o.src_pos] for o in ops if o.tag == "delete")
    ins = collections.Counter(hyp_n[o.dest_pos] for o in ops if o.tag == "insert")
    del_runs = runs(ops, "delete", 10)
    ins_runs = runs(ops, "insert", 10)

    latencies = [r["latency"] for r in results]
    langs = collections.Counter(r["language"] or "?" for r in results)

    samples = []
    if args.translate and results:
        translator = create_translator(cfg.translate)
        step = max(1, len(results) // args.translate)
        for r in results[::step][: args.translate]:
            if not r["text"]:
                continue
            out = translator.translate(
                TranslationRequest(
                    text=r["text"],
                    source_language=r["language"] or args.language,
                    target_language=cfg.translate.target_language,
                )
            )
            samples.append(
                {"source": r["text"], "english": out.text, "latency": round(out.latency, 3)}
            )

    report = {
        "audio_minutes": round(duration / 60, 2),
        "language_setting": args.language or "auto",
        "segments": len(segments),
        "speech_minutes": round(speech / 60, 2),
        "raw": raw,
        "reading": reading,
        "deletion_runs_ge10": {"count": len(del_runs), "chars": sum(del_runs)},
        "insertion_runs_ge10": {"count": len(ins_runs), "chars": sum(ins_runs)},
        "top_substitutions": [[a, b, c] for (a, b), c in subs.most_common(15)],
        "top_deleted": dels.most_common(15),
        "top_inserted": ins.most_common(15),
        "detected_languages": dict(langs),
        "asr_seconds": round(asr_time, 1),
        "real_time_factor": round(asr_time / duration, 3),
        "latency_mean": round(statistics.mean(latencies), 3) if latencies else 0,
        "latency_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3)
        if latencies
        else 0,
        "translation_samples": samples,
    }

    out = args.out or args.audio.with_name(
        f"{args.audio.stem}_results_{args.language or 'auto'}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audio.with_name(f"{args.audio.stem}_hyp_{args.language or 'auto'}.txt").write_text(
        "\n".join(r["text"] for r in results), encoding="utf-8"
    )
    (out.with_suffix(".segments.json")).write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print()
    print(f"raw CER      : {pct(raw['cer'])}  (accuracy {pct(1 - raw['cer'])})")
    print(f"reading CER  : {pct(reading['cer'])}  (accuracy {pct(1 - reading['cer'])})")
    print(
        f"errors       : {raw['substitutions']} sub / {raw['deletions']} del / "
        f"{raw['insertions']} ins over {raw['ref_chars']} ref chars"
    )
    print(f"missed runs  : {len(del_runs)} runs of >=10 chars ({sum(del_runs)} chars)")
    print(f"extra runs   : {len(ins_runs)} runs of >=10 chars ({sum(ins_runs)} chars)")
    print(f"languages    : {dict(langs)}")
    print(
        f"speed        : RTF {report['real_time_factor']} | latency mean "
        f"{report['latency_mean']}s p95 {report['latency_p95']}s"
    )
    print(f"report       : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
