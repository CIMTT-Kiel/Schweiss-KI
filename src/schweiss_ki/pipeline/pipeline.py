"""
Pipeline - End-to-End Orchestrierung
AP2.1

Phase 1: CAD-Konvertierung + WeldVolumeModel       ← abgeschlossen
Phase 2: + Preprocessing                           ← aktiv
Phase 3: + Segmentierung (RANSAC)                  ← noch nicht aktiv
Phase 4: + Segmentierung (PointNet, optional)      ← noch nicht aktiv
"""
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import open3d as o3d

from ..core.data_structures import WeldVolumeModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────

@dataclass
class CADConversionConfig:
    """Konfiguration für CAD-Konvertierung"""
    enabled: bool = True
    point_density: Optional[float] = None  # None = API-Standard


@dataclass
class PreprocessingConfig:
    """
    Konfiguration für Preprocessing (Phase 2).

    Die detaillierte Step-Konfiguration (Filter-Parameter, source_type_overrides)
    lebt in der pipeline.yaml unter dem 'preprocessing.steps'-Block und wird
    von PreprocessingPipeline.from_config() direkt gelesen.

    Hier liegt nur der globale Ein-/Ausschalter.
    """
    enabled: bool = False


@dataclass
class SegmentationConfig:
    """Konfiguration für Segmentierung (Phase 3/4)"""
    enabled: bool = False
    method: str = "ransac"          # ransac | pointnet | hybrid
    ransac_threshold: float = 0.25  # mm (Toleranzanforderung ±0.25mm)
    dbscan_eps: float = 0.5         # mm
    dbscan_min_points: int = 10


@dataclass
class OutputConfig:
    """Konfiguration für Output"""
    output_dir: Path = Path("data/processed")
    save_model: bool = True
    save_intermediate: bool = False  # Preprocessing-Zwischenschritte speichern


@dataclass
class PipelineConfig:
    """Hauptkonfiguration - wird aus YAML geladen"""
    input_dir: Path = Path("data/raw/step_files")
    output: OutputConfig = field(default_factory=OutputConfig)
    cad_conversion: CADConversionConfig = field(default_factory=CADConversionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Erzeugt PipelineConfig aus geparster YAML (dict)"""
        cfg = cls()

        if "input_dir" in d:
            cfg.input_dir = Path(d["input_dir"])

        if "output" in d:
            o = d["output"]
            cfg.output.output_dir = Path(o.get("output_dir", cfg.output.output_dir))
            cfg.output.save_model = o.get("save_model", cfg.output.save_model)
            cfg.output.save_intermediate = o.get("save_intermediate", cfg.output.save_intermediate)

        if "cad_conversion" in d:
            c = d["cad_conversion"]
            cfg.cad_conversion.enabled = c.get("enabled", cfg.cad_conversion.enabled)
            cfg.cad_conversion.point_density = c.get("point_density", cfg.cad_conversion.point_density)

        if "preprocessing" in d:
            p = d["preprocessing"]
            cfg.preprocessing.enabled = p.get("enabled", cfg.preprocessing.enabled)

        if "segmentation" in d:
            s = d["segmentation"]
            cfg.segmentation.enabled = s.get("enabled", cfg.segmentation.enabled)
            cfg.segmentation.method = s.get("method", cfg.segmentation.method)
            cfg.segmentation.ransac_threshold = s.get("ransac_threshold", cfg.segmentation.ransac_threshold)
            cfg.segmentation.dbscan_eps = s.get("dbscan_eps", cfg.segmentation.dbscan_eps)

        return cfg


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class Pipeline:
    """
    End-to-End Pipeline: STEP → WeldVolumeModel

    Phase 1: CAD-Konvertierung + Persistierung
    Phase 2: + Preprocessing (PreprocessingPipeline aus pipeline.yaml)
    Phase 3: + Segmentierung (folgt)
    """

    def __init__(self, config: PipelineConfig, config_path: Optional[Path] = None):
        self.config = config
        # config_path wird für PreprocessingPipeline.from_config() benötigt
        self._config_path = Path(config_path) if config_path else None
        self._setup_cad_converter()

    def _setup_cad_converter(self):
        """CAD Converter initialisieren"""
        try:
            from client.core import CADConverterClient
            self._cad_client = CADConverterClient()
            logger.info("CAD Converter initialisiert")
        except ImportError:
            logger.error("CAD API Client nicht gefunden – ist das Submodule korrekt eingerichtet?")
            raise

    # ── Haupt-Methoden ────────────────────────

    def process_file(self, step_file: Path) -> WeldVolumeModel:
        """
        Verarbeitet eine einzelne STEP-Datei → WeldVolumeModel

        Args:
            step_file: Pfad zur STEP-Datei

        Returns:
            WeldVolumeModel (ggf. gespeichert je nach Output-Config)
        """
        step_file = Path(step_file)
        model_id = step_file.stem
        logger.info(f"Verarbeite: {step_file.name}")
        t_start = time.time()

        # Stage 1: CAD-Konvertierung
        pcd = self._run_cad_conversion(step_file)

        # Stage 2: Preprocessing
        preprocessing_report = None
        if self.config.preprocessing.enabled:
            pcd, preprocessing_report = self._run_preprocessing(pcd, source_type="ideal")

        # WeldVolumeModel erstellen
        model = WeldVolumeModel(
            model_id=model_id,
            source_type="ideal",
            source_file=step_file,
            point_cloud=pcd,
            preprocessing_report=preprocessing_report,
        )

        # Stage 3: Segmentierung
        if self.config.segmentation.enabled:
            self._run_segmentation(model)

        # Ausgabe
        if self.config.output.save_model:
            save_path = model.save(self.config.output.output_dir)
            logger.info(f"  → Gespeichert: {save_path}")

        elapsed = time.time() - t_start
        logger.info(f"  ✓ {model_id}: {model.n_points:,} Punkte ({elapsed:.1f}s)")

        return model

    def process_scan(self, scan_file: Path, source_type: str = "real") -> WeldVolumeModel:
        """
        Verarbeitet eine Scan-Datei (PCD/PLY/XYZ) → WeldVolumeModel

        Args:
            scan_file: Pfad zur Scan-Datei
            source_type: "real" oder "synthetic"

        Returns:
            WeldVolumeModel
        """
        scan_file = Path(scan_file)
        model_id = scan_file.stem
        logger.info(f"Verarbeite Scan: {scan_file.name}")
        t_start = time.time()

        pcd = o3d.io.read_point_cloud(str(scan_file))
        if len(pcd.points) == 0:
            raise ValueError(f"Punktwolke leer oder Format nicht unterstützt: {scan_file}")
        logger.debug(f"  Scan geladen: {len(pcd.points):,} Punkte")

        preprocessing_report = None
        if self.config.preprocessing.enabled:
            pcd, preprocessing_report = self._run_preprocessing(pcd, source_type=source_type)

        model = WeldVolumeModel(
            model_id=model_id,
            source_type=source_type,
            source_file=scan_file,
            point_cloud=pcd,
            preprocessing_report=preprocessing_report,
        )

        if self.config.segmentation.enabled:
            self._run_segmentation(model)

        if self.config.output.save_model:
            save_path = model.save(self.config.output.output_dir)
            logger.info(f"  → Gespeichert: {save_path}")

        elapsed = time.time() - t_start
        logger.info(f"  ✓ {model_id}: {model.n_points:,} Punkte ({elapsed:.1f}s)")

        return model

    def process_directory(self, input_dir: Path = None) -> List[WeldVolumeModel]:
        """
        Verarbeitet alle STEP-Dateien in einem Verzeichnis (Batch)

        Args:
            input_dir: Verzeichnis mit STEP-Dateien (default: config.input_dir)

        Returns:
            Liste von WeldVolumeModels
        """
        input_dir = Path(input_dir or self.config.input_dir)
        step_files = (
            sorted(input_dir.glob("*.step")) +
            sorted(input_dir.glob("*.STEP")) +
            sorted(input_dir.glob("*.stp")) +
            sorted(input_dir.glob("*.STP"))
        )

        if not step_files:
            logger.warning(f"Keine STEP-Dateien gefunden in: {input_dir}")
            return []

        logger.info(f"Batch: {len(step_files)} Dateien in {input_dir}")
        t_batch_start = time.time()

        models = []
        errors = []

        for i, step_file in enumerate(step_files, 1):
            logger.info(f"[{i}/{len(step_files)}] {step_file.name}")
            try:
                model = self.process_file(step_file)
                models.append(model)
            except Exception as e:
                logger.error(f"  ✗ Fehler bei {step_file.name}: {e}")
                errors.append((step_file, e))

        elapsed = time.time() - t_batch_start
        logger.info(
            f"\nBatch abgeschlossen: {len(models)}/{len(step_files)} erfolgreich "
            f"({elapsed:.1f}s gesamt)"
        )
        if errors:
            logger.warning(f"{len(errors)} Fehler:")
            for f, e in errors:
                logger.warning(f"  - {f.name}: {e}")

        return models

    # ── Interne Stage-Methoden ─────────────────

    def _run_cad_conversion(self, step_file: Path) -> o3d.geometry.PointCloud:
        """Stage 1: STEP → Point Cloud via Michel's API"""
        tmp_ply = (
            self.config.output.output_dir
            / step_file.stem
            / "pointcloud.ply"
        )
        tmp_ply.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"  CAD Konvertierung: {step_file} → {tmp_ply}")
        ply_path = self._cad_client.convert_to_ply(str(step_file), str(tmp_ply))

        pcd = o3d.io.read_point_cloud(str(ply_path))
        logger.debug(f"  Point Cloud geladen: {len(pcd.points):,} Punkte")
        return pcd

    def _run_preprocessing(
        self,
        pcd: o3d.geometry.PointCloud,
        source_type: str,
    ) -> tuple:
        """
        Stage 2: Preprocessing via PreprocessingPipeline.

        Liest die Step-Konfiguration direkt aus der pipeline.yaml
        (Abschnitt 'preprocessing.steps' + 'preprocessing.source_type_overrides').

        Returns:
            Tuple aus (verarbeitete PointCloud, PreprocessingReport)
        """
        from ..preprocessing import PreprocessingPipeline

        if self._config_path is None:
            logger.warning(
                "Kein config_path gesetzt – Preprocessing mit Default-Parametern."
                " Pipeline mit config_path initialisieren für YAML-Konfiguration."
            )
            preprocessing_pipeline = PreprocessingPipeline(source_type=source_type)
        else:
            preprocessing_pipeline = PreprocessingPipeline.from_config(
                self._config_path,
                source_type=source_type,
            )

        logger.debug(f"  Preprocessing: {preprocessing_pipeline}")
        pcd_clean, report = preprocessing_pipeline.process(pcd)

        logger.debug(
            f"  Preprocessing abgeschlossen: "
            f"{report.points_in:,} → {report.points_out:,} Punkte "
            f"({report.total_retention_rate:.1%} behalten, "
            f"{report.total_duration_ms:.0f}ms)"
        )
        return pcd_clean, report

    def _run_segmentation(self, model: WeldVolumeModel):
        """Stage 3: Segmentierung (Phase 4 – Platzhalter)"""
        logger.debug("  Segmentierung: noch nicht implementiert (Phase 4)")