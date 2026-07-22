#!/usr/bin/env python3
"""
Sanity-Check für CoarsePCA: bekannte Transform anwenden → wiederfinden.

Workflow:
    1. Lade CAD-Punktwolke
    2. Mache eine Kopie als "Scan", wende bekannte Transform T_truth an
    3. Lasse CoarsePCA die Transform schätzen
    4. Vergleiche Schätzung mit Wahrheit, melde Fehler

Aufruf:
    uv run python scripts/test_coarse_pca.py
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path

import numpy as np
import open3d as o3d

from schweiss_ki.subtraction.registration import CoarsePCA, RegistrationPipeline
from schweiss_ki.core.console import force_utf8_output


# ─────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────

def build_rigid_transform(
    translation_mm: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    center: np.ndarray | None = None,
) -> np.ndarray:
    """4×4 Transform: Rotation (Rz @ Ry @ Rx) um center, dann translation."""
    rx, ry, rz = (math.radians(a) for a in rotation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    if center is None:
        T[:3, 3] = translation_mm
    else:
        T[:3, 3] = (np.eye(3) - R) @ center + np.array(translation_mm)
    return T


def transform_error(T_estimated: np.ndarray, T_truth_inv: np.ndarray) -> tuple[float, float]:
    """Fehler von T_estimated gegenüber der wahren Inversen.

    Returns:
        (rotation_error_deg, translation_error_mm)
    """
    T_err = T_estimated @ np.linalg.inv(T_truth_inv)
    R_err = T_err[:3, :3]
    # Rotation-Winkel aus Spur (trace) der Fehler-Rotation
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    rot_err_deg = math.degrees(math.acos(cos_angle))
    trans_err_mm = float(np.linalg.norm(T_err[:3, 3]))
    return rot_err_deg, trans_err_mm


# ─────────────────────────────────────────────────────────────────────────
# Test-Fälle
# ─────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # (name, translation_mm, rotation_deg)
    ("01_translation_small",     (0.5, 1.0, 0.2),   (0.0, 0.0, 0.0)),
    ("02_translation_large",     (5.0, 10.0, 2.0),  (0.0, 0.0, 0.0)),
    ("03_rotation_small",        (0.0, 0.0, 0.0),   (1.0, 0.5, 2.0)),
    ("04_rotation_large",        (0.0, 0.0, 0.0),   (10.0, 5.0, 20.0)),
    ("05_combined",              (2.0, 3.0, 0.5),   (3.0, 2.0, 5.0)),
    ("06_flip_180_z",            (0.0, 0.0, 0.0),   (0.0, 0.0, 180.0)),
    ("07_flip_180_x",            (0.0, 0.0, 0.0),   (180.0, 0.0, 0.0)),
]


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    force_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cad-ply",
        type=Path,
        default=Path(
            "data/processed/test_files/Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt/pointcloud.ply"
        ),
        help="Pfad zur CAD-Punktwolke.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Detailliertes Logging (Kandidaten-Kosten etc.)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.cad_ply.exists():
        print(f"❌ CAD-Datei nicht gefunden: {args.cad_ply}")
        return 1

    print(f"Lade CAD: {args.cad_ply}")
    cad = o3d.io.read_point_cloud(str(args.cad_ply))
    print(f"  {len(cad.points):,} Punkte")
    if len(cad.points) > 500_000:
        print(f"  → Downsample auf ~100k Punkte (für schnellere Tests)")
        # Voxel downsample für moderate Größe
        cad = cad.voxel_down_sample(voxel_size=0.5)
        print(f"  {len(cad.points):,} Punkte nach Downsampling")

    bbox = cad.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    print(f"  BBox: {bbox.get_extent()}, Center: {center}")
    print()

    # Pipeline mit nur einem Step
    pipeline = RegistrationPipeline([CoarsePCA(evaluation_samples=3000)])

    results = []
    for name, trans, rot in TEST_CASES:
        # Bekannte Transform
        T_truth = build_rigid_transform(trans, rot, center=center)
        T_truth_inv = np.linalg.inv(T_truth)

        # "Scan" erzeugen
        scan = copy.deepcopy(cad)
        scan.transform(T_truth)

        # Registrierung
        _aligned, report = pipeline.run(scan, cad)
        T_est = report.final_transform

        # Fehler
        rot_err, trans_err = transform_error(T_est, T_truth_inv)

        # Ausgabe
        flip = report.steps[0].artifacts.get("flip_chosen", "?")
        residual = report.steps[0].residual
        print(
            f"{name:30s}  "
            f"rot_err={rot_err:6.2f}°  "
            f"trans_err={trans_err:6.3f} mm  "
            f"residual={residual:.3f} mm  "
            f"flip={flip}"
        )

        results.append((name, rot_err, trans_err))

    # Zusammenfassung
    print()
    print("=" * 80)
    rot_errs = [r[1] for r in results]
    trans_errs = [r[2] for r in results]
    print(
        f"Rotation-Fehler:    mean={np.mean(rot_errs):.2f}°  max={max(rot_errs):.2f}°"
    )
    print(
        f"Translation-Fehler: mean={np.mean(trans_errs):.3f} mm  max={max(trans_errs):.3f} mm"
    )

    # Bewertung: für plate-like geometry sollte PCA <5° und <1mm liefern
    # bei sauberen Achsen, sonst muss ICP nachhelfen
    print()
    print("Hinweise zur Interpretation:")
    print("  - Rotation-Fehler < 5°: ICP-bereit (typische Konvergenzregion)")
    print("  - Rotation-Fehler > 30°: Vorzeichen-Ambiguität nicht aufgelöst – Bug")
    print("  - 180°-Rotation-Tests prüfen, dass die Ambiguitätsauflösung greift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())