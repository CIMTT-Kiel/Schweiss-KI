"""Reale XYZ-Scans (Scanner-Export) in .ply für die Pipeline konvertieren.

Das Scanner-Format ist eine Textdatei mit einer Kopfzeile (##-getrennt) und
Datenzeilen `X Y Z nx ny nz intensity` — die Normalen sind 0 (nicht gesetzt),
die 7. Spalte ein Qualitäts-/Intensitätswert. Wir nehmen nur X/Y/Z; Normalen
schätzt die Pipeline im Preprocessing (real-Pfad). Rohdaten bleiben unangetastet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

# Scanner-Dateiname → sauberer, ASCII-freundlicher model_id.
NAME_MAP = {
    "SCHWEIßSPALT 1,5 Spalt I": "real_spalt_1v5_I",
    "SCHWEIßSPALT 1,5 Spalt II": "real_spalt_1v5_II",
    "SCHWEIßSPALT O-Spalt": "real_spalt_0",       # Sollwert: kein Spalt
    "Schweißspalt 1,0 auf 2,5": "real_spalt_1v0_auf_2v5",
}


def load_xyz(path: Path) -> np.ndarray:
    """X/Y/Z aus dem Scanner-XYZ lesen; Kopf-/Fehlzeilen überspringen."""
    pts = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    return np.asarray(pts, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/raw/real_testscans"))
    ap.add_argument("--dst", type=Path, default=Path("data/raw/real_scans"))
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.src.iterdir()
                   if p.suffix.lower() == ".xyz")
    if not files:
        print(f"Keine .xyz-Dateien in {args.src}")
        return

    for f in files:
        pts = load_xyz(f)
        model_id = NAME_MAP.get(f.stem, f.stem.replace(" ", "_").replace(",", "v"))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        out = args.dst / f"{model_id}.ply"
        o3d.io.write_point_cloud(str(out), pcd)
        print(f"  {f.name:32s} -> {out.name:28s} {len(pts):7d} Punkte")

    print(f"OK: {len(files)} Scans konvertiert nach {args.dst}")


if __name__ == "__main__":
    main()
