#!/usr/bin/env python3
"""Erzeugt den Demofall C_ZX_01: Werkstück B um Z gedreht und in X verschoben.

Beide Fehler leben in der Draufsicht — die Z-Drehung öffnet den Spalt
keilförmig, die X-Verschiebung wirkt an den Enden; kein Höhenanteil, der erst
in einer 3D-Ansicht sichtbar wäre. Für das Präsentationsbild
`docs/figures/praesentation/02_drei_ebenen.png`.

Die Drehung erfolgt um den **Schwerpunkt von B**, nicht um den Weltursprung wie
im Standard-Generator. Dadurch bleibt die gemessene Relativlage sauber auf
Rz + Tx, ohne den Kopplungs-Ty, den eine Ursprungsdrehung erzeugt.

Aufruf (erzeugt den Scan, dann durch die Pipeline schicken):
    uv run python scripts/generate_rztx_demo.py
    uv run python scripts/run_batch_subtraction.py \\
        --scan-dir data/raw/_rztx_demo --source-type synthetic

Ergebnis: data/outputs/C_ZX_01/ (subtraction_report.json, pointcloud.ply, …)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from schweiss_ki.core.console import force_utf8_output
from generate_synthetic_dataset import (filter_top_surface, load_cad,
                                        split_workpieces)

RZ_DEG, TX_MM = 1.0, 3.0
NAME = "C_ZX_01"
REPO = Path(__file__).resolve().parents[1]
CAD_DIR = REPO / "data" / "outputs" / "cad" / "Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt"
OUT_DIR = REPO / "data" / "raw" / "_rztx_demo"


def main() -> int:
    force_utf8_output()
    cad = filter_top_surface(load_cad(CAD_DIR))
    cad_a, cad_b = split_workpieces(cad, split_axis=1, split_value=0.0)

    b = np.asarray(cad_b.points)
    c = b.mean(axis=0)
    t = np.radians(RZ_DEG)
    rz = np.array([[np.cos(t), -np.sin(t), 0],
                   [np.sin(t),  np.cos(t), 0],
                   [0, 0, 1]])
    b_new = (b - c) @ rz.T + c + np.array([TX_MM, 0.0, 0.0])

    cad_b_t = o3d.geometry.PointCloud(cad_b)
    cad_b_t.points = o3d.utility.Vector3dVector(b_new)
    merged = cad_a + cad_b_t

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{NAME}.ply"
    o3d.io.write_point_cloud(str(path), merged)
    print(f"geschrieben: {path}  ({len(merged.points):,} Punkte)  "
          f"Rz={RZ_DEG}° Tx={TX_MM} mm")
    print("weiter mit: uv run python scripts/run_batch_subtraction.py "
          "--scan-dir data/raw/_rztx_demo --source-type synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
