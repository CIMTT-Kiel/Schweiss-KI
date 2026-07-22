#!/usr/bin/env python3
"""
Batch-Verarbeitung: Alle Scans in einem Verzeichnis gegen ein CAD-Modell.

Aufruf:
    uv run python scripts/run_batch_subtraction.py
    uv run python scripts/run_batch_subtraction.py --scan-dir data/raw/cmm_scans \\
        --cad data/raw/step_files/Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt.STEP

Output:
    - Pro Scan: data/output/{scan_id}/pointcloud.ply + metadata.json
                                    + labels.npy + subtraction_report.json
                                    + gap_profile.png
    - Zusammenfassungstabelle als Print-Output und als CSV unter
      data/output/batch_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import yaml

from schweiss_ki.core.console import force_utf8_output
from schweiss_ki.pipeline.pipeline import Pipeline, PipelineConfig


def main():
    force_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("data/raw/cmm_scans"),
        help="Verzeichnis mit Scan-Dateien.",
    )
    parser.add_argument(
        "--cad",
        type=Path,
        default=Path("data/raw/step_files/Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt.STEP"),
        help="Pfad zur CAD-STEP-Datei.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Pfad zur pipeline.yaml.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.xyz", "*.XYZ", "*.ply", "*.PLY", "*.pcd", "*.PCD"],
        help="Glob-Patterns für Scan-Suche.",
    )
    parser.add_argument(
        "--source-type",
        choices=["real", "synthetic"],
        default="real",
        help="Steuert die Preprocessing-Overrides aus der pipeline.yaml. "
             "'synthetic' schaltet das Preprocessing ab – synthetische Scans "
             "sind rauschfrei und bringen exakte CAD-Normalen mit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Basis-Seed fuer reproduzierbare RANSAC-Ergebnisse. "
             "Ueberschreibt random_seed aus der Config. Mehrere Seeds fahren, "
             "um die Streuung gegen die RANSAC-Wahl zu quantifizieren.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Debug-Level-Logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ── Sanity Checks ────────────────────────────────────────────────
    if not args.scan_dir.exists():
        print(f"❌ Scan-Verzeichnis nicht gefunden: {args.scan_dir}")
        return 1
    if not args.cad.exists():
        print(f"❌ CAD-Datei nicht gefunden: {args.cad}")
        return 1
    if not args.config.exists():
        print(f"❌ Config nicht gefunden: {args.config}")
        return 1

    cfg = PipelineConfig.from_dict(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    if args.seed is not None:
        cfg.random_seed = args.seed
    logger = logging.getLogger(__name__)
    logger.info(f"Basis-Seed: {cfg.random_seed}")

    if not cfg.subtraction.enabled:
        print("❌ subtraction.enabled=false in der yaml – Subtraktion nicht aktiv.")
        return 1

    # ── Batch ausführen ──────────────────────────────────────────────
    pipeline = Pipeline(cfg, config_path=args.config)
    models = pipeline.process_scan_directory_against_cad(
        scan_dir=args.scan_dir,
        cad_step_file=args.cad,
        source_type=args.source_type,
        glob_patterns=tuple(args.patterns),
    )

    if not models:
        print("❌ Keine Bauteile verarbeitet.")
        return 1

    # ── Zusammenfassungstabelle ──────────────────────────────────────
    print()
    print("=" * 110)
    print("Zusammenfassung Subtraktions-Batch")
    print("=" * 110)
    header = (
        f"{'Bauteil':<30} {'Reg-Res':>9} | {'Spalt min':>10} {'Spalt max':>10} "
        f"{'Spalt mean':>11} {'Spalt std':>10} {'inlier':>7} {'Bins':>6}"
    )
    print(header)
    print("-" * 110)

    rows = []
    for m in models:
        if not m.has_subtraction:
            print(f"{m.model_id:<40} {'(no subtraction)':>56}")
            continue

        reg = m.subtraction_report.registration
        reg_res = (
            f"{reg.final_residual:.3f} mm"
            if reg.final_residual is not None else "n/a"
        )

        gp = m.subtraction_report.deviation.gap_profile
        if gp is None:
            row = (m.model_id, reg_res) + ("-",) * 6 + ("0/0",)
            print(f"{m.model_id:<30} {reg_res:>9} {'(no gap_profile)':>46}")
            rows.append(row)
            continue

        def stats(key):
            """min/max/mean/std eines Profil-Arrays, '-' wenn leer."""
            arr = gp.get(key)
            if arr is None:
                return ("-",) * 4 + (0, 0)
            a = np.asarray(arr, dtype=float)
            n_valid, n_total = int((~np.isnan(a)).sum()), len(a)
            if n_valid == 0:
                return ("-",) * 4 + (0, n_total)
            return (
                f"{np.nanmin(a):.3f}", f"{np.nanmax(a):.3f}",
                f"{np.nanmean(a):.3f}", f"{np.nanstd(a):.3f}",
                n_valid, n_total,
            )

        r_min, r_max, r_mean, r_std, n_valid, n_total = stats("gap_root_widths")

        ref = gp.get("reference_plane") or {}
        inlier = ref.get("inlier_ratio")
        inlier_s = f"{inlier:.3f}" if inlier is not None else "-"
        anchored = bool(gp.get("anchored"))

        row = (
            m.model_id, reg_res,
            r_min, r_max, r_mean, r_std,
            inlier_s, anchored, f"{n_valid}/{n_total}",
        )
        print(
            f"{m.model_id:<30} {reg_res:>9} | {r_min:>10} {r_max:>10} "
            f"{r_mean:>11} {r_std:>10} {inlier_s:>7} {n_valid}/{n_total:>3}"
        )
        rows.append(row)

    print("=" * 110)

    # ── CSV-Export ───────────────────────────────────────────────────
    csv_path = cfg.output.output_dir / "batch_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id", "reg_residual_mm",
            "gap_root_min_mm", "gap_root_max_mm", "gap_root_mean_mm",
            "gap_root_std_mm", "ref_plane_inlier_ratio", "anchored",
            "bins_valid",
        ])
        writer.writerows(rows)
    print(f"\n✓ {len(models)} Bauteile verarbeitet")
    print(f"  → Tabelle: {csv_path}")
    print(f"  → Plots:   {cfg.output.output_dir}/<model_id>/gap_profile.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())