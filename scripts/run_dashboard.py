#!/usr/bin/env python3
"""Startet das lokale Auswertungs-Dashboard.

    uv run python scripts/run_dashboard.py
    → http://127.0.0.1:8050

Optional: --outputs <dir> für ein anderes Batch-Verzeichnis, --port <n>.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from schweiss_ki.core.console import force_utf8_output
from schweiss_ki.dashboard import data
from schweiss_ki.dashboard.app import build_app


def main() -> int:
    force_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=data.DEFAULT_OUTPUTS,
                        help="Batch-Ausgabeverzeichnis (default: data/outputs).")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cases = data.discover_cases(args.outputs)
    if not cases:
        print(f"Keine Fälle mit Report in {args.outputs} gefunden.",
              file=sys.stderr)
        return 1
    print(f"{len(cases)} Fälle gefunden. Dashboard: http://127.0.0.1:{args.port}")
    build_app(args.outputs).run(port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
