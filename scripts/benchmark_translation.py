"""Compare translation backends against human reference translations.

Uses the WMT22 general-task test sets (news, e-commerce, social and
conversational text with professional reference translations), fetched by
sacrebleu on first use, and scores each translator with BLEU and chrF:

    python scripts/benchmark_translation.py \
        --translator qwen3-4b=llamacpp@http://127.0.0.1:8081 \
        --translator index-2b=index@http://127.0.0.1:8082 \
        --pairs ja-en,zh-en --sentences 200

Each translator is given isolated sentences, with no conversation history,
which is the standard sentence-level setup. Requires: pip install sacrebleu
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtsubs.config import TranslateConfig  # noqa: E402
from rtsubs.languages import canonical_name  # noqa: E402
from rtsubs.translate import TranslationRequest, create_translator  # noqa: E402


def load_pair(pair: str, sentences: int) -> tuple[list[str], list[str]]:
    """An evenly spaced sample of WMT22 source lines and their references."""

    def echo(side: str) -> list[str]:
        out = subprocess.run(
            [sys.executable, "-m", "sacrebleu", "-t", "wmt22", "-l", pair, "--echo", side],
            capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
        return out.splitlines()

    src, ref = echo("src"), echo("ref")
    step = max(1, len(src) // sentences)
    picks = list(range(0, len(src), step))[:sentences]
    return [src[i] for i in picks], [ref[i] for i in picks]


def run(label: str, backend: str, url: str, pairs: dict) -> dict:
    import sacrebleu

    translator = create_translator(TranslateConfig(backend=backend, base_url=url))
    ok, detail = translator.health()
    if not ok:
        raise SystemExit(f"{label}: {detail}")

    # Warm up so the first timed request doesn't include a cold start.
    translator.translate(TranslationRequest("Hello.", "English", "Japanese"))

    report: dict = {"backend": backend, "url": url, "pairs": {}}
    for pair, (src, ref) in pairs.items():
        source_lang, target_lang = (canonical_name(code) for code in pair.split("-"))
        hyps, latencies = [], []
        for line in src:
            started = time.perf_counter()
            out = translator.translate(TranslationRequest(line, source_lang, target_lang))
            latencies.append(time.perf_counter() - started)
            hyps.append(out.text)
        report["pairs"][pair] = {
            "sentences": len(src),
            "bleu": round(sacrebleu.corpus_bleu(hyps, [ref]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [ref]).score, 2),
            "latency_mean": round(statistics.mean(latencies), 3),
            "latency_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
            "empty_outputs": sum(1 for h in hyps if not h.strip()),
            "outputs": hyps,
        }
        print(
            f"  {label:12s} {pair}: BLEU {report['pairs'][pair]['bleu']:5.2f} | "
            f"chrF {report['pairs'][pair]['chrf']:5.2f} | "
            f"{report['pairs'][pair]['latency_mean'] * 1000:4.0f} ms/line",
            flush=True,
        )
    translator.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--translator",
        action="append",
        required=True,
        metavar="LABEL=BACKEND@URL",
        help="e.g. qwen3-4b=llamacpp@http://127.0.0.1:8081 (repeat to compare)",
    )
    ap.add_argument("--pairs", default="ja-en,zh-en")
    ap.add_argument("--sentences", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "translation_results.json")
    args = ap.parse_args()

    pairs = {p: load_pair(p, args.sentences) for p in args.pairs.split(",")}
    print(", ".join(f"{p}: {len(s)} sentences" for p, (s, _) in pairs.items()))

    results = {}
    for spec in args.translator:
        label, rest = spec.split("=", 1)
        backend, url = rest.split("@", 1)
        results[label] = run(label, backend, url, pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
