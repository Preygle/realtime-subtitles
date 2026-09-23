"""Entry point.

    python -m rtsubs                      # GUI: overlay + control panel
    python -m rtsubs --console            # no GUI, print subtitles to stdout
    python -m rtsubs --devices            # list capture devices and exit
    python -m rtsubs --check              # probe the model servers and exit
    python -m rtsubs --wav clip.wav --asr mock --translate passthrough
    python -m rtsubs --console --save srt,txt   # also save subtitles + transcript
    python -m rtsubs --file movie.mkv           # subtitle a file, then exit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, AppConfig
from .util.logging import setup_logging

log = logging.getLogger("rtsubs")


def _force_utf8_console() -> None:
    """Windows consoles default to cp1252, which cannot print CJK subtitles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtsubs",
        description="Realtime speech subtitles: transcribe, translate, overlay.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--console", action="store_true", help="run headless, printing to stdout"
    )
    parser.add_argument(
        "--devices", action="store_true", help="list capture devices and exit"
    )
    parser.add_argument(
        "--check", action="store_true", help="probe backends and exit"
    )
    parser.add_argument(
        "--file",
        type=Path,
        metavar="MEDIA",
        help="subtitle a video/audio file as fast as possible and exit",
    )
    parser.add_argument("--out", type=Path, help="--file: where to write (default: next to the media)")
    parser.add_argument("--wav", type=Path, help="replay a WAV file instead of a device")
    parser.add_argument("--loop-wav", action="store_true", help="loop the WAV file")

    parser.add_argument("--source", choices=("loopback", "input"))
    parser.add_argument("--device", help="substring of the capture device name")
    parser.add_argument("--asr", choices=("llamacpp", "whispercpp", "mock"))
    parser.add_argument("--asr-url")
    parser.add_argument(
        "--no-whisper-translate",
        action="store_true",
        help="whispercpp: transcribe only, then use the translation stage",
    )
    parser.add_argument("--language", help='spoken language, or "" to auto-detect')
    parser.add_argument(
        "--translate", dest="translate_backend",
        choices=("llamacpp", "nllb", "passthrough"),
    )
    parser.add_argument("--translate-url")
    parser.add_argument("--target", help="subtitle language (default: English)")
    parser.add_argument("--vad", choices=("auto", "silero", "energy"))
    parser.add_argument(
        "--save",
        metavar="FORMATS",
        help="save finished lines as any of srt,vtt,txt (comma-separated), e.g. --save srt,txt",
    )
    parser.add_argument("--save-dir", help="folder for saved transcripts (default: transcripts)")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    if args.source:
        cfg.audio.source = args.source
    if args.device is not None:
        cfg.audio.device_name = args.device
    if args.asr:
        cfg.asr.backend = args.asr
    if args.asr_url:
        if cfg.asr.backend == "whispercpp":
            cfg.asr.whisper_base_url = args.asr_url
        else:
            cfg.asr.base_url = args.asr_url
    if args.no_whisper_translate:
        cfg.asr.whisper_translate = False
    if args.language is not None:
        cfg.asr.language = args.language
    if args.translate_backend:
        cfg.translate.backend = args.translate_backend
    if args.translate_url:
        cfg.translate.base_url = args.translate_url
    if args.target:
        cfg.translate.target_language = args.target
    if args.vad:
        cfg.vad.engine = args.vad
    if args.log_level:
        cfg.log_level = args.log_level
    if args.save:
        from .export import FORMATS

        wanted = [f.strip().lower().lstrip(".") for f in args.save.split(",") if f.strip()]
        unknown = [f for f in wanted if f not in FORMATS]
        if unknown:
            raise SystemExit(f"--save: unknown format(s) {unknown}; choose from {list(FORMATS)}")
        cfg.export.enabled = True
        cfg.export.formats = wanted
    if args.save_dir:
        cfg.export.directory = args.save_dir


def cmd_devices() -> int:
    from .audio.devices import available_backends, list_devices

    backends = available_backends()
    if not backends:
        print("No capture backend installed.")
        print("  pip install PyAudioWPatch   (preferred, WASAPI loopback)")
        print("  pip install soundcard       (pure-Python fallback)")
        return 1

    print(f"Capture backends: {', '.join(backends)}\n")
    devices = list_devices()
    if not devices:
        print("No capture devices found.")
        return 1
    for device in devices:
        default = "  <- default" if device.is_default else ""
        print(
            f"  {device.label}\n"
            f"      backend={device.backend} channels={device.channels} "
            f"rate={device.sample_rate}{default}"
        )
    return 0


def cmd_file(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Subtitle one media file, showing progress on a single line."""
    from .offline import MediaError, subtitle_file

    def human(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    last = [0.0]

    def on_progress(p) -> None:
        now = time.monotonic()
        if p.fraction < 1.0 and now - last[0] < 0.25:
            return
        last[0] = now
        bar_width = 28
        filled = int(bar_width * p.fraction)
        eta = f" ETA {human(p.eta_seconds)}" if p.eta_seconds > 1 else ""
        print(
            f"\r  [{'#' * filled}{'.' * (bar_width - filled)}] {p.fraction * 100:5.1f}%  "
            f"{human(p.media_position)}/{human(p.media_seconds)}{eta}   ",
            end="",
            file=sys.stderr,
            flush=True,
        )

    print(f"Subtitling {args.file.name}", file=sys.stderr, flush=True)
    try:
        result = subtitle_file(
            args.file,
            cfg,
            output_dir=args.out or args.save_dir,
            formats=cfg.export.formats,
            on_progress=on_progress,
        )
    except MediaError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print(file=sys.stderr)

    if not result.lines:
        print("No speech found.", file=sys.stderr)
        return 1
    languages = ", ".join(
        f"{name} ({count})" for name, count in sorted(result.languages.items(), key=lambda kv: -kv[1])
    )
    print(
        f"{result.lines} lines from {human(result.media_seconds)} of media "
        f"in {human(result.elapsed_seconds)} ({result.speed:.1f}x real time)\n"
        f"Detected: {languages}",
        file=sys.stderr,
    )
    for path in result.outputs.values():
        print(path)
    return 0


def cmd_check(cfg: AppConfig) -> int:
    from .asr import create_asr
    from .audio.vad import create_vad
    from .translate import create_translator

    ok = True
    for label, factory, config in (
        ("ASR", create_asr, cfg.asr),
        ("Translator", create_translator, cfg.translate),
    ):
        try:
            backend = factory(config)
            reachable, detail = backend.health()
            print(f"{label:11s} [{'OK ' if reachable else 'FAIL'}] {detail}")
            ok = ok and reachable
            backend.close()
        except Exception as exc:
            print(f"{label:11s} [FAIL] {exc}")
            ok = False

    try:
        vad = create_vad(cfg.vad)
        print(f"{'VAD':11s} [OK ] {vad.name}")
    except Exception as exc:
        print(f"{'VAD':11s} [FAIL] {exc}")
        ok = False

    from .audio.dsp import resampler_name

    print(f"{'Resampler':11s} [OK ] {resampler_name()}")
    if cfg.asr.backend == "whispercpp" and cfg.asr.whisper_translate:
        print(
            f"{'Mode':11s} [OK ] single-pass: Whisper outputs English directly "
            "(translation stage bypassed)"
        )
    else:
        print(
            f"{'Mode':11s} [OK ] two-stage: {cfg.asr.backend} -> "
            f"{cfg.translate.backend} -> {cfg.translate.target_language}"
        )
    return 0 if ok else 1


def cmd_console(cfg: AppConfig, args: argparse.Namespace) -> int:
    from .audio.capture import WavFileSource
    from .pipeline import SubtitlePipeline

    source = None
    if args.wav:
        if not args.wav.is_file():
            print(f"WAV file not found: {args.wav}", file=sys.stderr)
            return 1
        source = WavFileSource(str(args.wav), loop=args.loop_wav)

    def on_event(event) -> None:
        if event.is_final:
            print(
                f"[{event.language or '?':10s}] {event.text}"
                f"   ({event.total_latency * 1000:.0f} ms)",
                flush=True,
            )

    def on_status(level: str, message: str) -> None:
        print(f"  ({level}) {message}", file=sys.stderr, flush=True)

    pipeline = SubtitlePipeline(cfg, on_event, on_status, source=source)
    try:
        pipeline.start()
    except Exception as exc:
        print(f"Failed to start: {exc}", file=sys.stderr)
        return 1

    print("Listening. Press Ctrl+C to stop.", file=sys.stderr, flush=True)
    try:
        while True:
            time.sleep(0.2)
            if source is not None and not source.running:
                # WAV replay finished; drain the last segment.
                time.sleep(1.5)
                break
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    finally:
        pipeline.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)

    cfg = AppConfig.load(args.config)
    apply_overrides(cfg, args)
    setup_logging(cfg.log_level, PROJECT_ROOT / "logs" / "rtsubs.log")

    if args.devices:
        return cmd_devices()
    if args.file:
        return cmd_file(cfg, args)
    if args.check:
        return cmd_check(cfg)
    if args.console or args.wav:
        return cmd_console(cfg, args)

    try:
        from .ui.app import run_app
    except ImportError as exc:
        print(f"GUI unavailable ({exc}). Install PySide6, or use --console.", file=sys.stderr)
        return 1
    return run_app(cfg, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
