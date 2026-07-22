#!/usr/bin/env python3
"""
Synthetischen Datensatz generieren: Ground-Truth-Transformationen von
Werkstück B relativ zu Werkstück A, um die Detektion in ComponentRegistration
und PointDistance zu validieren.

Vorgehen:
    1. CAD-Punktwolke aus dem Cache laden
    2. Auf top-surface filtern (n_z > 0.5) – simuliert Scan von oben
    3. In Werkstück A (Y ≥ 0) und B (Y < 0) trennen
    4. Für jeden Testfall: Werkstück B mit definierter Transformation
       versehen, mit A zusammenführen, als PLY speichern
    5. Zentrale CSV mit allen eingebrachten Parametern schreiben

Aufruf:
    uv run python scripts/generate_synthetic_dataset.py

Output:
    data/raw/synthetic_scans/synthetic_metadata.csv
    data/raw/synthetic_scans/<filename>.ply
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d


# ── Test-Cases ──────────────────────────────────────────────────────────

@dataclass
class TestCase:
    filename: str
    category: str
    tx_mm: float = 0.0
    ty_mm: float = 0.0
    tz_mm: float = 0.0
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0
    description: str = ""


def _t_name(prefix: str, axis: str, value: float) -> str:
    return f"{prefix}_{axis}_{value:+07.3f}mm"


def _r_name(prefix: str, axis: str, value: float) -> str:
    return f"{prefix}_{axis}_{value:+07.3f}deg"


def build_test_cases() -> List[TestCase]:
    """Alle Testfälle nach Vorgabe."""
    cases: List[TestCase] = []

    # ── Translation X (Naht-Längsrichtung, symmetrisch – nur positive Werte)
    for v in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
        cases.append(TestCase(
            filename=_t_name("T", "X", v),
            category="translation_x",
            tx_mm=v,
            description=f"Werkstück B in X um {v:+.3f} mm verschoben",
        ))

    # ── Translation Y (Spalt-Querrichtung, symmetrisch mit Vorzeichen)
    for v in [-1.5, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 1.5]:
        cases.append(TestCase(
            filename=_t_name("T", "Y", v),
            category="translation_y",
            ty_mm=v,
            description=f"Werkstück B in Y um {v:+.3f} mm verschoben "
                        f"(Spalt {'enger' if v < 0 else 'weiter'})",
        ))

    # ── Translation Z (vertikal)
    for v in [0.1, 0.25, 0.5, 1.0]:
        cases.append(TestCase(
            filename=_t_name("T", "Z", v),
            category="translation_z",
            tz_mm=v,
            description=f"Werkstück B in Z um {v:+.3f} mm verschoben (Höhenversatz)",
        ))

    # ── Translation Kombinationen (5 Fälle)
    trans_combos = [
        (0.25, 0.25, 0.1,  "kleiner Versatz alle Achsen"),
        (0.5,  0.5,  0.1,  "mittlerer Versatz alle Achsen"),
        (1.0,  -0.5, 0.25, "X-Versatz mit Spaltverengung und Z"),
        (2.0,  1.0,  0.5,  "größerer Versatz alle Achsen"),
        (3.0,  -1.0, 0.5,  "Extremfall alle Achsen"),
    ]
    for i, (tx, ty, tz, desc) in enumerate(trans_combos, 1):
        cases.append(TestCase(
            filename=f"C_TT_{i:02d}",
            category="translation_combo",
            tx_mm=tx, ty_mm=ty, tz_mm=tz,
            description=desc,
        ))

    # ── Rotation X (Aufklappen/Schließen, symmetrisch mit Vorzeichen)
    for v in [-2.0, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 2.0]:
        cases.append(TestCase(
            filename=_r_name("R", "X", v),
            category="rotation_x",
            rx_deg=v,
            description=f"Rotation um X um {v:+.3f}° "
                        f"({'Werkstücke aufgeklappt' if v > 0 else 'zusammengeklappt'})",
        ))

    # ── Rotation Y (Wurzel wandert in Z entlang Naht)
    for v in [0.1, 0.25, 0.5, 1.0]:
        cases.append(TestCase(
            filename=_r_name("R", "Y", v),
            category="rotation_y",
            ry_deg=v,
            description=f"Rotation um Y um {v:+.3f}° (Wurzel wandert entlang X)",
        ))

    # ── Rotation Z (Spalt an einem Ende enger, am anderen weiter)
    for v in [0.1, 0.25, 0.5, 1.0]:
        cases.append(TestCase(
            filename=_r_name("R", "Z", v),
            category="rotation_z",
            rz_deg=v,
            description=f"Rotation um Z um {v:+.3f}° (Spalt keilförmig)",
        ))

    # ── Rotation Kombinationen (5 Fälle)
    rot_combos = [
        (0.25, 0.1,  0.1,  "kleine Rotation alle Achsen"),
        (0.5,  0.25, 0.25, "mittlere Rotation alle Achsen"),
        (1.0,  0.5,  0.5,  "größere Rotation alle Achsen"),
        (-0.5, 0.25, 0.25, "negatives rot_x + kleine y/z"),
        (-1.0, 0.5,  1.0,  "Extremfall mit negativem rot_x"),
    ]
    for i, (rx, ry, rz, desc) in enumerate(rot_combos, 1):
        cases.append(TestCase(
            filename=f"C_RR_{i:02d}",
            category="rotation_combo",
            rx_deg=rx, ry_deg=ry, rz_deg=rz,
            description=desc,
        ))

    # ── Translation + Rotation Kombinationen (12 Fälle) ─ typische Fehlerbilder
    full_combos = [
        # Szenario: leicht versetzt und leicht verkippt
        (0.5, 0.2, 0.0,  0.25, 0.0, 0.0, "klein: leicht versetzt + verkippt"),
        (1.0, 0.5, 0.0,  0.5,  0.0, 0.0, "mittel: leicht versetzt + verkippt"),
        (2.0, 1.0, 0.0,  1.0,  0.0, 0.0, "groß: leicht versetzt + verkippt"),

        # Szenario: Spalt öffnen mit Verdrehung
        (0.0, 1.0, 0.0,  0.0,  0.0, 0.5, "klein: Spalt weiter + verdreht"),
        (0.0, 1.5, 0.0,  0.0,  0.0, 1.0, "mittel: Spalt weiter + verdreht"),

        # Szenario: Höhenversatz mit Verkippung entlang Naht
        (0.0, 0.0, 0.25, 0.0,  0.25, 0.0, "klein: Höhenversatz + Verkippung"),
        (0.0, 0.0, 0.5,  0.0,  0.5,  0.0, "mittel: Höhenversatz + Verkippung"),
        (0.0, 0.0, 1.0,  0.0,  1.0,  0.0, "groß: Höhenversatz + Verkippung"),

        # Szenario: vollständige 6-DOF Abweichung
        (0.5, 0.3, 0.1,  0.25, 0.1,  0.1, "klein: 6-DOF Abweichung"),
        (1.0, 0.5, 0.25, 0.5,  0.25, 0.25, "mittel: 6-DOF Abweichung"),
        (2.0, -1.0, 0.5, 1.0,  0.5,  0.5, "groß: 6-DOF Abweichung"),
        (3.0, 1.0, 0.5,  -1.0, 0.5,  -1.0, "extrem: 6-DOF Abweichung"),
    ]
    for i, (tx, ty, tz, rx, ry, rz, desc) in enumerate(full_combos, 1):
        cases.append(TestCase(
            filename=f"C_TR_{i:02d}",
            category="translation_rotation_combo",
            tx_mm=tx, ty_mm=ty, tz_mm=tz,
            rx_deg=rx, ry_deg=ry, rz_deg=rz,
            description=desc,
        ))

    return cases


# ── Transformation ──────────────────────────────────────────────────────

def build_transformation(
    tx: float, ty: float, tz: float,
    rx_deg: float, ry_deg: float, rz_deg: float,
) -> np.ndarray:
    """Baut 4×4 Transformation. R = R_x · R_y · R_z, dann Translation.

    Konvention identisch zu ComponentRegistration – Ground-Truth-Werte
    lassen sich direkt mit ausgelesenen Euler-Winkeln vergleichen.
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [ 0,          1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])
    R = Rx @ Ry @ Rz
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


# ── Hauptlogik ──────────────────────────────────────────────────────────

def load_cad(cad_cache_dir: Path) -> o3d.geometry.PointCloud:
    """Lädt die CAD-Punktwolke aus dem Konvertierungs-Cache."""
    ply_path = cad_cache_dir / "pointcloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(
            f"CAD-Cache-PLY nicht gefunden: {ply_path}\n"
            "Erst run_batch_subtraction.py laufen lassen, damit CAD konvertiert ist."
        )
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_normals():
        raise ValueError(
            f"CAD-Punktwolke {ply_path} hat keine Normalen – "
            "Cache scheint korrupt."
        )
    return pcd


def filter_top_surface(pcd: o3d.geometry.PointCloud, nz_threshold: float = 0.5):
    """Entfernt Punkte mit nach unten oder seitlich zeigenden Normalen.

    Simuliert die Sichtbarkeit eines CMM-Scans von oben – nur Punkte,
    deren Oberflächen-Normale hauptsächlich nach oben zeigt (n_z > 0.5),
    werden übernommen.

    TODO: nz_threshold=0.5 ist für flachere V-Nähte zu streng. Eine Flanke
    mit Winkel α zur Vertikalen hat n_z = sin(α):
        90° V-Naht (α=45°) → n_z = 0.707  → passiert den Filter
        60° V-Naht (α=30°) → n_z = 0.500  → liegt exakt auf der Schwelle,
                                            die Flanken würden komplett
                                            herausgefiltert.
    Vor dem Erzeugen von 60°-Datensätzen den Schwellwert an den Flanken-
    winkel koppeln (z.B. nz_threshold = 0.5 * sin(α)). Bewusst noch nicht
    geändert, um die bestehenden 61 Fälle nicht zu invalidieren.
    """
    normals = np.asarray(pcd.normals)
    mask = normals[:, 2] > nz_threshold
    return pcd.select_by_index(np.where(mask)[0].tolist())


def split_workpieces(
    pcd: o3d.geometry.PointCloud,
    split_axis: int = 1, split_value: float = 0.0,
):
    """Trennt Punktwolke in Werkstück A (positive Seite) und B (negative)."""
    points = np.asarray(pcd.points)
    mask_a = points[:, split_axis] >= split_value
    pcd_a = pcd.select_by_index(np.where(mask_a)[0].tolist())
    pcd_b = pcd.select_by_index(np.where(~mask_a)[0].tolist())
    return pcd_a, pcd_b


def generate_case(
    cad_a: o3d.geometry.PointCloud,
    cad_b: o3d.geometry.PointCloud,
    case: TestCase,
    output_dir: Path,
) -> Path:
    """Erzeugt Punktwolke für einen Testfall und speichert als PLY.

    Konvention der Metadaten (case.ty_mm):
        +ty > 0  ->  Werkstücke werden auseinander geschoben (Spalt öffnet sich)
        -ty < 0  ->  Werkstücke werden zusammen geschoben (Spalt schließt sich)

    Werkstück B liegt bei Y < 0. Um "positive ty = Öffnung" zu erreichen,
    muss B in -Y-Richtung wandern. Daher wird ty beim Anwenden auf B
    invertiert. Die Metadaten in der CSV bleiben in User-Konvention.
    """
    T = build_transformation(
        tx=case.tx_mm,
        ty=-case.ty_mm,         # Invertierung: siehe Docstring
        tz=case.tz_mm,
        rx_deg=case.rx_deg,
        ry_deg=case.ry_deg,
        rz_deg=case.rz_deg,
    )

    cad_b_transformed = o3d.geometry.PointCloud(cad_b)
    cad_b_transformed.transform(T)

    merged = cad_a + cad_b_transformed
    output_path = output_dir / f"{case.filename}.ply"
    o3d.io.write_point_cloud(str(output_path), merged)
    return output_path


def write_metadata_csv(cases: List[TestCase], output_dir: Path) -> Path:
    """Schreibt zentrale Übersicht als CSV."""
    csv_path = output_dir / "synthetic_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "category",
            "tx_mm", "ty_mm", "tz_mm",
            "rx_deg", "ry_deg", "rz_deg",
            "description",
        ])
        for c in cases:
            writer.writerow([
                f"{c.filename}.ply", c.category,
                f"{c.tx_mm:.4f}", f"{c.ty_mm:.4f}", f"{c.tz_mm:.4f}",
                f"{c.rx_deg:.4f}", f"{c.ry_deg:.4f}", f"{c.rz_deg:.4f}",
                c.description,
            ])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cad-cache-dir",
        type=Path,
        default=Path("data/outputs/cad/Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt"),
        help="Verzeichnis mit dem konvertierten CAD (pointcloud.ply darin).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/synthetic_scans"),
        help="Zielverzeichnis für die synthetischen Bauteile.",
    )
    parser.add_argument(
        "--no-top-filter", action="store_true",
        help="CAD-Punktwolke NICHT auf top-surface filtern (n_z > 0.5).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    if not args.cad_cache_dir.exists():
        logger.error(f"❌ CAD-Cache-Verzeichnis nicht gefunden: {args.cad_cache_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CAD laden und vorbereiten
    logger.info(f"CAD laden: {args.cad_cache_dir}")
    cad = load_cad(args.cad_cache_dir)
    logger.info(f"  Ursprünglich: {len(cad.points):,} Punkte")

    if not args.no_top_filter:
        cad = filter_top_surface(cad)
        logger.info(f"  Nach Top-Surface-Filter (n_z > 0.5): {len(cad.points):,} Punkte")

    cad_a, cad_b = split_workpieces(cad, split_axis=1, split_value=0.0)
    logger.info(
        f"  Werkstück A (Y≥0): {len(cad_a.points):,} pts | "
        f"Werkstück B (Y<0): {len(cad_b.points):,} pts"
    )

    # Testfälle erzeugen
    cases = build_test_cases()
    logger.info(f"Erzeuge {len(cases)} Testfälle …")

    # Gruppierung für Übersicht im Log
    by_category: dict = {}
    for c in cases:
        by_category.setdefault(c.category, []).append(c)

    for cat, items in by_category.items():
        logger.info(f"  {cat}: {len(items)}")

    for i, case in enumerate(cases, 1):
        out_path = generate_case(cad_a, cad_b, case, args.output_dir)
        if i == 1 or i % 10 == 0 or i == len(cases):
            logger.info(f"  [{i:3d}/{len(cases)}] {out_path.name}")

    # Metadaten-CSV schreiben
    csv_path = write_metadata_csv(cases, args.output_dir)
    logger.info(f"✓ Metadaten-CSV: {csv_path}")
    logger.info(f"✓ Synthetischer Datensatz erstellt in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())