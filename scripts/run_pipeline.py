#!/usr/bin/env python3
"""
run_pipeline.py - Schweiß-KI Pipeline Runner

Verwendung:

    # venv aktivieren
    source .venv/bin/activate
    # Einzelne Datei
    python scripts/run_pipeline.py --input data/raw/step_files/model.step

    # Ganzes Verzeichnis (Batch)
    python scripts/run_pipeline.py --batch

    # Eigene Config
    python scripts/run_pipeline.py --config configs/phase1.yaml --batch

    # Config überschreiben (z.B. anderen Output-Ordner)
    python scripts/run_pipeline.py --batch --output-dir data/processed/experiment_01
"""
import argparse
import logging
from pathlib import Path

import yaml

from schweiss_ki.pipeline.pipeline import Pipeline, PipelineConfig


# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────
# Config laden
# ─────────────────────────────────────────────

def load_config(config_path: Path, overrides: dict) -> PipelineConfig:
    """Lädt YAML-Config und wendet CLI-Overrides an"""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # CLI-Overrides einarbeiten
    if overrides.get("input_dir"):
        raw["input_dir"] = overrides["input_dir"]
    if overrides.get("output_dir"):
        raw.setdefault("output", {})["output_dir"] = overrides["output_dir"]

    return PipelineConfig.from_dict(raw)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Schweiß-KI Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Modus
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input", "-i",
        type=Path,
        metavar="FILE",
        help="Einzelne STEP-Datei verarbeiten",
    )
    mode.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Alle STEP-Dateien in input_dir verarbeiten",
    )

    # Config
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        metavar="CONFIG",
        help="Pfad zur YAML-Konfiguration (default: configs/pipeline.yaml)",
    )

    # Overrides
    parser.add_argument(
        "--input-dir",
        type=str,
        metavar="DIR",
        help="Überschreibt input_dir aus Config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="DIR",
        help="Überschreibt output.output_dir aus Config",
    )

    # Optionen
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Debug-Logging aktivieren",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Config laden und validieren, ohne tatsächlich zu konvertieren",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Config laden
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config nicht gefunden: {config_path}")
        sys.exit(1)

    overrides = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
    }
    config = load_config(config_path, overrides)

    # Info ausgeben
    logger.info("=" * 60)
    logger.info("Schweiß-KI Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config:          {config_path}")
    logger.info(f"Input:           {config.input_dir}")
    logger.info(f"Output:          {config.output.output_dir}")
    logger.info(f"Preprocessing:   {'aktiv' if config.preprocessing.enabled else 'deaktiviert'}")
    logger.info(f"Segmentierung:   {'aktiv' if config.segmentation.enabled else 'deaktiviert'}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("Dry-run: Config valide, kein Durchlauf.")
        sys.exit(0)

    # Pipeline starten
    pipeline = Pipeline(config)

    if args.input:
        # Einzeldatei
        if not args.input.exists():
            logger.error(f"Datei nicht gefunden: {args.input}")
            sys.exit(1)
        model = pipeline.process_file(args.input)
        logger.info(f"\nErgebnis: {model}")

    elif args.batch:
        # Batch
        models = pipeline.process_directory()
        logger.info(f"\n{len(models)} Modelle erfolgreich verarbeitet.")


if __name__ == "__main__":
    main()