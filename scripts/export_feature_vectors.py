"""AP3 — Featurevektoren exportieren: eine Zeile pro Bauteil aus data/outputs.

Scannt das Output-Verzeichnis (kein fest verdrahteter Fall-Katalog, wie das
Dashboard), baut je Bauteil die Merkmalszeile aus dem Subtraktions-Report und
schreibt sie als CSV und JSON. Läuft für synthetische wie echte Fälle.

Aufruf:
    uv run python scripts/export_feature_vectors.py
    uv run python scripts/export_feature_vectors.py --thickness 5 --reinforcement 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schweiss_ki.analysis.feature_vector import ALL_COLUMNS, build_feature_row
from schweiss_ki.dashboard import data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=data.DEFAULT_OUTPUTS,
                    help="Verzeichnis mit den Fall-Ausgaben.")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV-Zielpfad (Default: <outputs>/feature_vectors.csv).")
    ap.add_argument("--thickness", type=float, default=5.0,
                    help="Materialstärke in mm (Eingabe für das Volumen).")
    ap.add_argument("--reinforcement", type=float, default=0.0,
                    help="Nahtüberhöhung in mm (Aufschlag aufs Volumen).")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    out_csv = args.out or outputs / "feature_vectors.csv"
    cases = data.discover_cases(outputs)
    if not cases:
        print(f"Keine Fälle in {outputs} gefunden.")
        return

    rows = []
    for case in cases:
        report = data.load_report(str(outputs), case)
        src = data.source_type(str(outputs), case)
        rows.append(build_feature_row(
            report, case, src,
            thickness_mm=args.thickness, reinforcement_mm=args.reinforcement))

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    df.to_csv(out_csv, index=False)
    out_json = out_csv.with_suffix(".json")
    df.to_json(out_json, orient="records", indent=2, force_ascii=False)

    print(f"OK: {len(df)} Featurevektoren geschrieben")
    print(f"  -> {out_csv}")
    print(f"  -> {out_json}")
    print(f"  Spalten: {len(ALL_COLUMNS)} "
          f"(Identitaet {4}, Merkmale {len(ALL_COLUMNS) - 4 - 17}, Diagnose {17})")
    n_syn = (df["source_type"] == "synthetic").sum()
    print(f"  Fälle: {n_syn} synthetisch, {len(df) - n_syn} real")


if __name__ == "__main__":
    main()
