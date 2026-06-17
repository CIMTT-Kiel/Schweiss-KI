#!/usr/bin/env python3
"""
Smoke-Test für die integrierte Subtraktions-Pipeline.

Workflow:
    1. Lädt pipeline.yaml
    2. Verarbeitet einen Scan gegen ein CAD-Modell via
       process_scan_against_cad()
    3. Prüft, ob das Ergebnis (subtraction_report) sinnvolle Werte enthält

Erwartung: gleiche Werte wie aus dem Notebook
    - Registrierungs-Residuum ~0.18 mm
    - GapProfile detektiert variable Spaltbreite

Aufruf:
    uv run python scripts/smoke_test_subtraction.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from schweiss_ki.pipeline.pipeline import Pipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        type=Path,
        default=Path("data/raw/cmm_scans/SCHWEIßSPALT_1,0_auf_2,5.xyz"),
        help="Pfad zur Scan-Datei.",
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
        "--verbose", "-v", action="store_true",
        help="Debug-Level-Logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Config laden
    if not args.config.exists():
        print(f"❌ Config nicht gefunden: {args.config}")
        return 1
    cfg = PipelineConfig.from_dict(yaml.safe_load(args.config.read_text()))

    # Sanity-Checks
    if not args.scan.exists():
        print(f"❌ Scan nicht gefunden: {args.scan}")
        return 1
    if not args.cad.exists():
        print(f"❌ CAD nicht gefunden: {args.cad}")
        return 1
    if not cfg.subtraction.enabled:
        print("❌ subtraction.enabled=false in der yaml – Subtraktion nicht aktiv.")
        return 1

    print("=" * 70)
    print("Smoke-Test: Subtraktions-Pipeline")
    print("=" * 70)
    print(f"Scan:   {args.scan}")
    print(f"CAD:    {args.cad}")
    print(f"Config: {args.config}")
    print()

    # Pipeline ausführen
    pipeline = Pipeline(cfg, config_path=args.config)
    model = pipeline.process_scan_against_cad(
        scan_file=args.scan,
        cad_step_file=args.cad,
        source_type="real",
    )

    # ── Verifikation ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Verifikation")
    print("=" * 70)

    if not model.has_subtraction:
        print("❌ FEHLER: subtraction_report wurde nicht gesetzt.")
        return 1

    sub = model.subtraction_report
    print(f"✓ subtraction_report vorhanden")
    print(f"  cad_source_file: {sub.cad_source_file}")
    print()

    # Registrierungs-Report
    reg = sub.registration
    print(f"Registrierung:")
    print(f"  Steps:            {len(reg.steps)}")
    print(f"  Gesamt-Laufzeit:  {reg.total_duration_ms:.1f} ms")
    print(f"  Final residual:   "
          f"{reg.final_residual:.3f} mm" if reg.final_residual is not None else "  Final residual:   n/a")
    print(f"  Final transform:  diag={np.diag(reg.final_transform)}")
    for step in reg.steps:
        res = f"residual={step.residual:.3f}mm" if step.residual is not None else "residual=n/a"
        fit = f", fitness={step.fitness:.3f}" if step.fitness is not None else ""
        print(f"    - {step.step_name}: {step.duration_ms:.1f} ms, {res}{fit}")
    print()

    # Deviation-Report
    dev = sub.deviation
    print(f"Differenzanalyse:")
    print(f"  Toleranz:         ±{dev.tolerance_mm} mm")
    print(f"  Gesamt-Laufzeit:  {dev.total_duration_ms:.1f} ms")
    print(f"  Steps:            {len(dev.step_reports)}")
    for step in dev.step_reports:
        print(f"    - {step.step_name}: {step.duration_ms:.1f} ms")
        if step.artifacts:
            for k, v in step.artifacts.items():
                if isinstance(v, float):
                    print(f"        {k}: {v:.3f}")
                else:
                    print(f"        {k}: {v}")
    print()

    # GapProfile (falls vorhanden)
    if dev.gap_profile is not None:
        gp = dev.gap_profile
        gw = gp["gap_widths"]
        valid = ~np.isnan(gw)
        n_valid = valid.sum()
        if n_valid > 0:
            print(f"GapProfile:")
            print(f"  Gültige Bins:     {n_valid}/{len(gw)}")
            print(f"  Spaltbreite:      min={np.nanmin(gw):.2f} mm, "
                  f"max={np.nanmax(gw):.2f} mm, mean={np.nanmean(gw):.2f} mm")

    print()
    print("=" * 70)
    print("✓ Smoke-Test erfolgreich")
    print("=" * 70)
    print()
    print(f"Modell gespeichert in: {cfg.output.output_dir / model.model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())