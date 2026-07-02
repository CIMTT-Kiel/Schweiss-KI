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

from schweiss_ki.pipeline.pipeline import Pipeline, PipelineConfig


def main():
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

    cfg = PipelineConfig.from_dict(yaml.safe_load(args.config.read_text()))
    if not cfg.subtraction.enabled:
        print("❌ subtraction.enabled=false in der yaml – Subtraktion nicht aktiv.")
        return 1

    # ── Batch ausführen ──────────────────────────────────────────────
    pipeline = Pipeline(cfg, config_path=args.config)
    models = pipeline.process_scan_directory_against_cad(
        scan_dir=args.scan_dir,
        cad_step_file=args.cad,
        source_type="real",
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
    header = f"{'Bauteil':<40} {'Reg-Res':>10} {'Gap min':>10} {'Gap max':>10} {'Gap mean':>10} {'Gap std':>10} {'Bins':>6}"
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
        if gp is None or "gap_widths" not in gp:
            row = (m.model_id, reg_res, "-", "-", "-", "-", "0/0")
            print(f"{m.model_id:<40} {reg_res:>10} {'(no gap_profile)':>46}")
        else:
            gw = np.asarray(gp["gap_widths"])
            valid = ~np.isnan(gw)
            n_valid = int(valid.sum())
            n_total = len(gw)
            if n_valid > 0:
                gmin = f"{np.nanmin(gw):.3f}"
                gmax = f"{np.nanmax(gw):.3f}"
                gmean = f"{np.nanmean(gw):.3f}"
                gstd = f"{np.nanstd(gw):.3f}"
            else:
                gmin = gmax = gmean = gstd = "-"
            row = (m.model_id, reg_res, gmin, gmax, gmean, gstd, f"{n_valid}/{n_total}")
            print(
                f"{m.model_id:<40} {reg_res:>10} "
                f"{gmin:>10} {gmax:>10} {gmean:>10} {gstd:>10} {n_valid}/{n_total:>3}"
            )
        rows.append(row)

    print("=" * 110)

    # ── CSV-Export ───────────────────────────────────────────────────
    csv_path = cfg.output.output_dir / "batch_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id", "reg_residual_mm",
            "gap_min_mm", "gap_max_mm", "gap_mean_mm", "gap_std_mm",
            "bins_valid",
        ])
        writer.writerows(rows)
    print(f"\n✓ {len(models)} Bauteile verarbeitet")
    print(f"  → Tabelle: {csv_path}")
    print(f"  → Plots:   {cfg.output.output_dir}/<model_id>/gap_profile.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())