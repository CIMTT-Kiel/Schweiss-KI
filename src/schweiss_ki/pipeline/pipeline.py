"""
Pipeline - End-to-End Orchestrierung
AP2.1 + AP2.2

Phase 1: CAD-Konvertierung + WeldVolumeModel       ← abgeschlossen
Phase 2: + Preprocessing                           ← aktiv
Phase 3: + Segmentierung (RANSAC)                  ← aktiv
Phase 4: + Segmentierung (PointNet, optional)      ← noch nicht aktiv
Phase 5: + Subtraktion (Registrierung + Deviation) ← aktiv
"""
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d

from ..core.data_structures import WeldVolumeModel
from schweiss_ki.segmentation import SegmentationPipeline, LABELS
from schweiss_ki.subtraction.deviation import DeviationPipeline
from schweiss_ki.subtraction.registration import RegistrationPipeline
from schweiss_ki.subtraction.reports import SubtractionReport

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────

@dataclass
class CADConversionConfig:
    """Konfiguration für CAD-Konvertierung.

    target_point_spacing: gewünschter mittlerer Punktabstand auf der CAD-
        Oberfläche in mm. Aus Bauteil-Oberfläche (analyse_cad) wird
        point_count = total_area / spacing² abgeleitet und an die API
        übergeben. None = API-Default-Sampling.
    """
    enabled: bool = True
    target_point_spacing: Optional[float] = None


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
class SubtractionConfig:
    """Konfiguration für Subtraktion (AP2.2).

    Globaler Ein-/Ausschalter. Die detaillierte Step-Konfiguration
    (registration.steps, deviation.steps) lebt in der pipeline.yaml und wird
    von RegistrationPipeline.from_config() bzw. DeviationPipeline.from_config()
    direkt gelesen.
    """
    enabled: bool = False
    cad_top_normal_z_threshold: float = 0.5  # n_z-Schwellwert für CAD-Top-Surface-Filter


@dataclass
class OutputConfig:
    """Konfiguration für Output"""
    output_dir: Path = Path("data/output")
    save_model: bool = True
    save_intermediate: bool = False
    save_subtraction_plots: bool = True


@dataclass
class PipelineConfig:
    """Hauptkonfiguration - wird aus YAML geladen"""
    input_dir: Path = Path("data/raw/step_files")
    output: OutputConfig = field(default_factory=OutputConfig)
    cad_conversion: CADConversionConfig = field(default_factory=CADConversionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    subtraction: SubtractionConfig = field(default_factory=SubtractionConfig)

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
            cfg.output.save_subtraction_plots = o.get(   # ← NEU
                "save_subtraction_plots", cfg.output.save_subtraction_plots
            )

        if "cad_conversion" in d:
            c = d["cad_conversion"]
            cfg.cad_conversion.enabled = c.get("enabled", cfg.cad_conversion.enabled)
            cfg.cad_conversion.target_point_spacing = c.get(
                "target_point_spacing", cfg.cad_conversion.target_point_spacing
            )

        if "preprocessing" in d:
            p = d["preprocessing"]
            cfg.preprocessing.enabled = p.get("enabled", cfg.preprocessing.enabled)

        if "segmentation" in d:
            s = d["segmentation"]
            cfg.segmentation.enabled = s.get("enabled", cfg.segmentation.enabled)
            cfg.segmentation.method = s.get("method", cfg.segmentation.method)
            cfg.segmentation.ransac_threshold = s.get("ransac_threshold", cfg.segmentation.ransac_threshold)
            cfg.segmentation.dbscan_eps = s.get("dbscan_eps", cfg.segmentation.dbscan_eps)

        if "subtraction" in d:
            sub = d["subtraction"]
            cfg.subtraction.enabled = sub.get("enabled", cfg.subtraction.enabled)
            cfg.subtraction.cad_top_normal_z_threshold = sub.get(
                "cad_top_normal_z_threshold",
                cfg.subtraction.cad_top_normal_z_threshold,
            )

        return cfg


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class Pipeline:
    """
    End-to-End Pipeline: STEP → WeldVolumeModel (+ optional Subtraktion gegen CAD)

    Phase 1: CAD-Konvertierung + Persistierung
    Phase 2: + Preprocessing
    Phase 3: + Segmentierung
    Phase 5: + Subtraktion (Scan ↔ CAD-Vergleich)
    """

    def __init__(self, config: PipelineConfig, config_path: Optional[Path] = None):
        self.config = config
        # config_path wird für PreprocessingPipeline/SegmentationPipeline/
        # RegistrationPipeline/DeviationPipeline.from_config() benötigt
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

    def process_scan(
        self,
        scan_file: Path,
        source_type: str = "real",
        cad_step_file: Optional[Path] = None,
    ) -> WeldVolumeModel:
        """
        Verarbeitet eine Scan-Datei (PCD/PLY/XYZ) → WeldVolumeModel

        Args:
            scan_file: Pfad zur Scan-Datei
            source_type: "real" oder "synthetic"
            cad_step_file: Optional. Wenn gesetzt und subtraction.enabled, wird
                der Scan zusätzlich gegen das angegebene CAD-STEP verglichen
                (Registrierung + Differenzanalyse). Das CAD wird automatisch
                konvertiert oder aus dem Cache geladen.

        Returns:
            WeldVolumeModel mit ggf. gesetztem subtraction_report
        """
        scan_file = Path(scan_file)
        model_id = scan_file.stem

        if cad_step_file is not None:
            logger.info(
                f"Verarbeite Scan vs. CAD: {scan_file.name} vs. {Path(cad_step_file).name}"
            )
        else:
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

        # Stage 4: Subtraktion gegen CAD
        if cad_step_file is not None and self.config.subtraction.enabled:
            cad_pcd = self._get_or_convert_cad(Path(cad_step_file))
            self._run_subtraction(model, cad_pcd, cad_source_file=Path(cad_step_file))
        elif cad_step_file is not None and not self.config.subtraction.enabled:
            logger.warning(
                "  cad_step_file gesetzt, aber subtraction.enabled=false – "
                "Subtraktion übersprungen."
            )

        if self.config.output.save_model:
            save_path = model.save(self.config.output.output_dir)
            logger.info(f"  → Gespeichert: {save_path}")

        elapsed = time.time() - t_start
        logger.info(f"  ✓ {model_id}: {model.n_points:,} Punkte ({elapsed:.1f}s)")

        return model

    def process_scan_against_cad(
        self,
        scan_file: Path,
        cad_step_file: Path,
        source_type: str = "real",
    ) -> WeldVolumeModel:
        """
        Verarbeitet einen Scan und vergleicht ihn gegen ein CAD-Modell.

        Convenience-Wrapper um `process_scan()` mit gesetztem cad_step_file.

        Args:
            scan_file: Pfad zur Scan-Datei
            cad_step_file: Pfad zur CAD-STEP-Datei
            source_type: "real" oder "synthetic"
        """
        return self.process_scan(
            scan_file,
            source_type=source_type,
            cad_step_file=cad_step_file,
        )

    def process_directory(self, input_dir: Path = None) -> List[WeldVolumeModel]:
        """
        Verarbeitet alle STEP-Dateien in einem Verzeichnis (Batch)

        Args:
            input_dir: Verzeichnis mit STEP-Dateien (default: config.input_dir)

        Returns:
            Liste von WeldVolumeModels
        """
        input_dir = Path(input_dir or self.config.input_dir)
        # Dedupliziert über den aufgelösten Pfad: auf case-insensitiven
        # Dateisystemen (Windows, macOS) matchen "*.step" und "*.STEP"
        # dieselbe Datei, die sonst doppelt verarbeitet würde.
        seen: set = set()
        step_files = []
        for pattern in ("*.step", "*.STEP", "*.stp", "*.STP"):
            for path in sorted(input_dir.glob(pattern)):
                key = path.resolve()
                if key not in seen:
                    seen.add(key)
                    step_files.append(path)

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

    def process_scans_against_cad(
        self,
        scan_files: List[Path],
        cad_step_file: Path,
        source_type: str = "real",
    ) -> List[WeldVolumeModel]:
        """
        Batch: mehrere Scans gegen ein einzelnes CAD-Modell.

        CAD wird einmalig konvertiert (oder aus dem Cache geladen) und dann
        für alle Scans wiederverwendet.

        Args:
            scan_files: Liste der Scan-Pfade
            cad_step_file: Pfad zur CAD-STEP-Datei
            source_type: "real" oder "synthetic"
        """
        cad_step_file = Path(cad_step_file)
        logger.info(
            f"Batch: {len(scan_files)} Scans gegen {cad_step_file.name}"
        )
        t_batch_start = time.time()

        # CAD einmal vorbereiten (Cache wird hier ggf. angelegt)
        self._get_or_convert_cad(cad_step_file)

        models = []
        errors = []
        for i, scan_file in enumerate(scan_files, 1):
            scan_file = Path(scan_file)
            logger.info(f"[{i}/{len(scan_files)}] {scan_file.name}")
            try:
                model = self.process_scan(
                    scan_file,
                    source_type=source_type,
                    cad_step_file=cad_step_file,
                )
                models.append(model)
            except Exception as e:
                logger.error(f"  ✗ Fehler bei {scan_file.name}: {e}")
                errors.append((scan_file, e))

        elapsed = time.time() - t_batch_start
        logger.info(
            f"\nBatch abgeschlossen: {len(models)}/{len(scan_files)} erfolgreich "
            f"({elapsed:.1f}s gesamt)"
        )
        if errors:
            logger.warning(f"{len(errors)} Fehler:")
            for f, e in errors:
                logger.warning(f"  - {f.name}: {e}")

        return models

    def process_scan_directory_against_cad(
        self,
        scan_dir: Path,
        cad_step_file: Path,
        source_type: str = "real",
        glob_patterns: tuple = ("*.xyz", "*.ply", "*.pcd"),
    ) -> List[WeldVolumeModel]:
        """
        Batch: alle Scans in einem Verzeichnis gegen ein CAD-Modell.

        Args:
            scan_dir: Verzeichnis mit Scan-Dateien
            cad_step_file: Pfad zur CAD-STEP-Datei
            source_type: "real" oder "synthetic"
            glob_patterns: Datei-Pattern für Scan-Suche
        """
        scan_dir = Path(scan_dir)
        scan_files: List[Path] = []
        for pattern in glob_patterns:
            scan_files.extend(scan_dir.glob(pattern))
        scan_files = sorted(set(scan_files))

        if not scan_files:
            logger.warning(
                f"Keine Scans gefunden mit Patterns {glob_patterns} in {scan_dir}"
            )
            return []

        return self.process_scans_against_cad(
            scan_files,
            cad_step_file,
            source_type=source_type,
        )

    # ── Interne Stage-Methoden ─────────────────

    def _run_cad_conversion(
        self,
        step_file: Path,
        output_ply: Optional[Path] = None,
    ) -> o3d.geometry.PointCloud:
        """Stage 1: STEP → Point Cloud via Michel's API.

        Wenn target_point_spacing gesetzt ist, wird die benötigte Punktanzahl
        aus der CAD-Oberfläche abgeleitet (analyse_cad → total_area).
        Bei Fehler in analyse_cad: Fallback auf API-Default.

        Args:
            step_file: Pfad zur STEP-Datei.
            output_ply: Optional. Zielpfad für die PLY-Datei. Default:
                {output_dir}/{step_stem}/pointcloud.ply.
        """
        if output_ply is None:
            output_ply = (
                self.config.output.output_dir
                / step_file.stem
                / "pointcloud.ply"
            )
        output_ply.parent.mkdir(parents=True, exist_ok=True)

        kwargs = {}
        spacing = self.config.cad_conversion.target_point_spacing
        if spacing is not None:
            try:
                analysis = self._cad_client.analyse_cad(str(step_file))
                total_area = sum(obj["surface_area"] for obj in analysis["objects"])  # mm²
                point_count = max(1, int(total_area / spacing ** 2))
                kwargs["point_count"] = point_count
                logger.info(
                    f"  CAD-Sampling: spacing={spacing} mm, "
                    f"area={total_area:.1f} mm² → point_count={point_count:,}"
                )
            except Exception as e:
                logger.warning(
                    f"  analyse_cad fehlgeschlagen ({type(e).__name__}: {e}). "
                    f"Fallback auf API-Default-Sampling."
                )

        logger.debug(f"  CAD Konvertierung: {step_file} → {output_ply}")
        ply_path = self._cad_client.convert_to_ply(
            str(step_file), str(output_ply), **kwargs
        )

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

    def _run_segmentation(self, model: WeldVolumeModel) -> None:
        """Stage 3: Segmentierung der vorverarbeiteten Punktwolke."""
        if self._config_path is None:
            logger.warning("  Keine config_path gesetzt, Segmentierung übersprungen")
            return

        seg_pipeline = SegmentationPipeline.from_config(self._config_path)
        if len(seg_pipeline.steps) == 0:
            logger.warning("  Keine Segmentation-Steps in Config aktiviert")
            return

        labels, report = seg_pipeline.process(model.point_cloud)
        model.labels = labels
        model.label_names = {int(k): v for k, v in LABELS.items()}
        model.segmentation_method = "ransac"

        logger.info(
            f"  Segmentierung: {len(seg_pipeline.steps)} Steps, "
            f"coverage {report.coverage_pct:.1f}%"
        )

    def _run_subtraction(
        self,
        scan_model: WeldVolumeModel,
        cad_pcd: o3d.geometry.PointCloud,
        cad_source_file: Optional[Path] = None,
    ) -> None:
        """Stage 4: Subtraktion - Vergleich Scan vs. CAD.

        Mutiert scan_model:
            - point_cloud wird in CAD-Koordinatensystem überführt
            - subtraction_report wird gesetzt

        Args:
            scan_model: Segmentierter Scan (idealerweise mit Labels).
            cad_pcd: CAD-Punktwolke mit Normalen.
            cad_source_file: Pfad zur Quell-STEP-Datei (für den Report).
        """
        if self._config_path is None:
            logger.warning("  Kein config_path gesetzt, Subtraktion übersprungen")
            return

        # CAD-Top-Surface-Filter für Registrierung (2.5D-Scan vs 3D-CAD)
        cad_for_registration = self._filter_cad_top_surface(cad_pcd)

        # Registrierung
        reg_pipeline = RegistrationPipeline.from_config(self._config_path)
        if len(reg_pipeline.steps) == 0:
            logger.warning("  Keine Registrierungs-Steps in Config aktiviert")
            return

        scan_aligned, reg_report = reg_pipeline.run(
            scan_model.point_cloud,
            cad_for_registration,
            source_labels=scan_model.labels,
            target_labels=None,
        )

        # Scan dauerhaft ins CAD-Koordinatensystem überführen
        scan_model.point_cloud = scan_aligned

        # Differenzanalyse auf der ausgerichteten Wolke (gegen das volle CAD)
        dev_pipeline = DeviationPipeline.from_config(self._config_path)
        if len(dev_pipeline.steps) > 0:
            dev_data = dev_pipeline.run(
                scan_aligned,
                cad_pcd,
                source_labels=scan_model.labels,
                target_labels=None,
            )
        else:
            from schweiss_ki.subtraction.reports import DeviationData
            dev_data = DeviationData()
            logger.warning("  Keine Deviation-Steps in Config – leerer DeviationData")

        # Report bauen
        scan_model.subtraction_report = SubtractionReport(
            registration=reg_report,
            deviation=dev_data,
            cad_source_file=str(cad_source_file) if cad_source_file else None,
        )
        
        # Subtraktions-Plots speichern (falls aktiviert)
        if self.config.output.save_subtraction_plots:
            self._save_subtraction_plots(scan_model)

        # Zusammenfassung
        final_res_str = (
            f"{reg_report.final_residual:.3f} mm"
            if reg_report.final_residual is not None else "n/a"
        )
        logger.info(
            f"  Subtraktion: Registrierung residual={final_res_str}, "
            f"{len(dev_pipeline.steps)} Deviation-Steps"
        )

    def _filter_cad_top_surface(
        self,
        cad_pcd: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:
        """Filtert CAD auf Punkte mit nach oben zeigender Normale.

        Hintergrund: Reale Scans erfassen typischerweise nur die von oben
        sichtbare Oberfläche eines Bauteils. Das CAD enthält das volle
        3D-Volumen inkl. Unterseite, Stirnseiten etc. Diese Asymmetrie
        verzerrt den PCA-Schwerpunkt bei der Grob-Registrierung – Lösung:
        CAD vor der Registrierung auf die "scanbare" Oberfläche reduzieren.

        Filter: n_z > cad_top_normal_z_threshold (Default 0.5, ~60° aus Vertikaler).
        """
        if not cad_pcd.has_normals():
            logger.warning(
                "  CAD ohne Normalen – Top-Surface-Filter nicht möglich, "
                "ungefiltertes CAD wird für Registrierung verwendet."
            )
            return cad_pcd

        threshold = self.config.subtraction.cad_top_normal_z_threshold
        normals = np.asarray(cad_pcd.normals)
        top_mask = normals[:, 2] > threshold
        idx = np.where(top_mask)[0]

        if len(idx) == 0:
            logger.warning(
                f"  CAD-Top-Filter (n_z > {threshold}) ergab 0 Punkte – "
                f"ungefiltertes CAD wird verwendet."
            )
            return cad_pcd

        cad_top = cad_pcd.select_by_index(idx.tolist())
        logger.info(
            f"  CAD-Top-Surface-Filter: {len(cad_pcd.points):,} → "
            f"{len(cad_top.points):,} Punkte (n_z > {threshold})"
        )
        return cad_top

    # ── CAD-Cache ─────────────────────────────

    def _get_cad_cache_dir(self, step_file: Path) -> Path:
        """Cache-Verzeichnis für eine CAD-STEP-Datei."""
        return self.config.output.output_dir / "cad" / step_file.stem

    def _is_cad_cache_valid(self, step_file: Path) -> bool:
        """Prüft, ob der CAD-Cache aktuell ist (STEP-Mtime + Konvertierungs-Parameter)."""
        cache_dir = self._get_cad_cache_dir(step_file)
        cache_info_file = cache_dir / "cache_info.json"
        pointcloud_file = cache_dir / "pointcloud.ply"

        if not (cache_info_file.exists() and pointcloud_file.exists()):
            return False

        try:
            with open(cache_info_file, encoding="utf-8") as f:
                info = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        # STEP-Mtime vergleichen
        current_mtime = step_file.stat().st_mtime
        if info.get("step_mtime") != current_mtime:
            logger.debug(
                f"  CAD-Cache invalidiert: STEP geändert "
                f"(cached={info.get('step_mtime')}, current={current_mtime})"
            )
            return False

        # Konvertierungs-Parameter vergleichen
        cached_spacing = info.get("target_point_spacing")
        current_spacing = self.config.cad_conversion.target_point_spacing
        if cached_spacing != current_spacing:
            logger.debug(
                f"  CAD-Cache invalidiert: target_point_spacing geändert "
                f"(cached={cached_spacing}, current={current_spacing})"
            )
            return False

        return True

    def _get_or_convert_cad(self, step_file: Path) -> o3d.geometry.PointCloud:
        """Lädt CAD aus Cache oder konvertiert es neu.

        Cache-Pfad: {output_dir}/cad/{step_stem}/
        Cache-Validierung: STEP-Mtime + target_point_spacing.
        """
        step_file = Path(step_file)
        if not step_file.exists():
            raise FileNotFoundError(f"CAD-STEP nicht gefunden: {step_file}")

        cache_dir = self._get_cad_cache_dir(step_file)

        if self._is_cad_cache_valid(step_file):
            logger.info(f"  CAD-Cache HIT: {cache_dir}")
            pcd = o3d.io.read_point_cloud(str(cache_dir / "pointcloud.ply"))
            return pcd

        logger.info(f"  CAD-Cache MISS, konvertiere: {step_file.name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        pcd = self._run_cad_conversion(
            step_file,
            output_ply=cache_dir / "pointcloud.ply",
        )

        # Cache-Info speichern
        cache_info = {
            "step_file": str(step_file),
            "step_mtime": step_file.stat().st_mtime,
            "target_point_spacing": self.config.cad_conversion.target_point_spacing,
            "n_points": len(pcd.points),
            "has_normals": pcd.has_normals(),
        }
        with open(cache_dir / "cache_info.json", "w", encoding="utf-8") as f:
            json.dump(cache_info, f, indent=2)

        return pcd

    def _save_subtraction_plots(self, model: WeldVolumeModel) -> None:
        """Speichert Plot-Visualisierungen des Subtraktions-Ergebnisses im Modell-Ordner."""
        from schweiss_ki.subtraction.plots import save_gap_profile_png

        plot_dir = self.config.output.output_dir / model.model_id
        plot_dir.mkdir(parents=True, exist_ok=True)

        gap_path = save_gap_profile_png(
            model,
            output_path=plot_dir / "gap_profile.png",
        )
        if gap_path:
            logger.info(f"  → Plot gespeichert: {gap_path}")