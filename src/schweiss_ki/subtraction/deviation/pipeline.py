"""
DeviationPipeline – orchestriert eine Sequenz von DeviationSteps.

Im Gegensatz zur Registrierung akkumulieren Deviation-Steps nicht eine
Transformation, sondern befüllen ein gemeinsames DeviationData-Objekt
(signed distances, per-region-metrics, gap profile, ...).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d

from ..base import DeviationStep
from ..reports import DeviationData

logger = logging.getLogger(__name__)


class DeviationPipeline:
    """Verkettete Ausführung mehrerer DeviationSteps.

    Nutzung:
        pipeline = DeviationPipeline(
            steps=[PointDistance(...), PerRegionMetrics(...), ToleranceClassifier(...)],
            tolerance_mm=0.25,
        )
        data = pipeline.run(source_aligned, target, source_labels, target_labels)
    """

    def __init__(
        self,
        steps: List[DeviationStep],
        tolerance_mm: float = 0.25,
    ):
        self.steps: List[DeviationStep] = list(steps)
        self.tolerance_mm = tolerance_mm

    # ── Ausführung ────────────────────────────────────────────────────

    def run(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> DeviationData:
        """Wendet alle Steps nacheinander an, jeder schreibt in `data`.

        Args:
            source:         Bereits ausgerichteter Scan.
            target:         CAD-Referenz.
            source_labels:  Optionale Labels für source (aus AP2.1).
            target_labels:  Optionale Labels für target.

        Returns:
            DeviationData mit allen Step-Beiträgen.
        """
        if len(source.points) == 0:
            raise ValueError("Source PointCloud ist leer.")
        if len(target.points) == 0:
            raise ValueError("Target PointCloud ist leer.")

        logger.info(
            f"DeviationPipeline: {len(self.steps)} Steps, "
            f"source={len(source.points):,} pts, target={len(target.points):,} pts, "
            f"tolerance=±{self.tolerance_mm} mm"
        )

        data = DeviationData(tolerance_mm=self.tolerance_mm)
        t0_total = time.perf_counter()

        for step in self.steps:
            if not step.enabled:
                logger.debug(f"  Skip (disabled): {step.name}")
                data.step_reports.append(
                    step(source, target, data, source_labels, target_labels)
                )
                continue

            logger.debug(f"  Run: {step.name}")
            report = step(source, target, data, source_labels, target_labels)
            data.step_reports.append(report)
            logger.info(f"  ✓ {step.name}: {report.duration_ms:.1f} ms")

        data.total_duration_ms = (time.perf_counter() - t0_total) * 1000.0
        logger.info(f"DeviationPipeline fertig: {data.total_duration_ms:.1f} ms gesamt")

        return data

    # ── Verkettung / Inspektion ───────────────────────────────────────

    def add(self, step: DeviationStep) -> "DeviationPipeline":
        self.steps.append(step)
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        names = ", ".join(s.name for s in self.steps)
        return f"DeviationPipeline([{names}], tol=±{self.tolerance_mm}mm)"

    # ── YAML-Loading ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: Path) -> "DeviationPipeline":
        """Lädt eine Pipeline aus dem Abschnitt 'subtraction.deviation.steps'
        der pipeline.yaml.

        TODO: Implementierung sobald konkrete Steps existieren.
              Pattern analog zu PreprocessingPipeline.from_config().
        """
        raise NotImplementedError(
            "DeviationPipeline.from_config() wird implementiert, "
            "sobald konkrete Steps existieren."
        )