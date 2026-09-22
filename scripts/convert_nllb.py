"""Convert NLLB-200 to CTranslate2 for the CPU translation backend.

    python scripts/convert_nllb.py
    python scripts/convert_nllb.py --model facebook/nllb-200-distilled-1.3B

Downloads roughly 2.5 GB for the 600M model, then writes an int8 CTranslate2
directory of about 600 MB into models/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quantization", default="int8")
    args = parser.parse_args()

    output = args.output or PROJECT_ROOT / "models" / (
        args.model.split("/")[-1] + "-ct2"
    )
    if (output / "model.bin").is_file():
        print(f"Already converted: {output}")
        return 0

    try:
        import ctranslate2  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print(
            "Missing dependencies. Install them with:\n"
            "  pip install ctranslate2 transformers sentencepiece",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", args.model,
        "--output_dir", str(output),
        "--quantization", args.quantization,
        "--force",
    ]
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    print(f"\nConverted to {output}")
    print('Set translate.backend to "nllb" in config.json, or run with')
    print("  python -m rtsubs --translate nllb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
