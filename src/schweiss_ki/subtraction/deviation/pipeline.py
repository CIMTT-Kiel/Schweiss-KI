"""
DeviationPipeline – orchestriert eine Sequenz von DeviationSteps.

Steps befüllen ein gemeinsames DeviationData-Objekt (signed distances,
per-region-metrics, voxel_deviation, component_registration, gap_profile).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d
import yaml

from ..base import DeviationStep
from ..reports import DeviationData

logger = logging.getLogger(__name__)


def _build_step_registry() -> Dict[str, type]:
    from .component_registration import ComponentRegistration
    from .gap_profile import GapProfile
    from .point_distance import PointDistance
    from .voxel_deviation import VoxelDeviation
    return {
        "point_distance": PointDistance,
        "voxel_deviation": VoxelDeviation,
        "component_registration": ComponentRegistration,
        "gap_profile": GapProfile,
    }


class DeviationPipeline:
    """Verkettete Ausführung mehrerer DeviationSteps."""

    def __init__(
        self,
        steps: List[DeviationStep],
        tolerance_mm: float = 0.25,
    ):
        self.steps: List[DeviationStep] = list(steps)
        self.tolerance_mm = float(tolerance_mm)

    def run(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> DeviationData:
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

    def add(self, step: DeviationStep) -> "DeviationPipeline":
        self.steps.append(step)
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        names = ", ".join(s.name for s in self.steps)
        return f"DeviationPipeline([{names}], tol=±{self.tolerance_mm}mm)"

    @classmethod
    def from_config(cls, config_path: Path) -> "DeviationPipeline":
        """Lädt Pipeline aus 'subtraction.deviation.steps' der pipeline.yaml.

        Reihenfolge der Schlüssel = Ausführungsreihenfolge. Wichtig:
        voxel_deviation braucht point_distance vorher.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        deviation_cfg = cfg.get("subtraction", {}).get("deviation", {})
        tolerance_mm = float(deviation_cfg.get("tolerance_mm", 0.25))
        steps_cfg = deviation_cfg.get("steps", {})

        if not steps_cfg:
            logger.warning(
                "Kein 'subtraction.deviation.steps'-Block in Config gefunden – "
                "Pipeline bleibt leer."
            )
            return cls(steps=[], tolerance_mm=tolerance_mm)

        registry = _build_step_registry()
        steps: List[DeviationStep] = []

        for step_name, step_params in steps_cfg.items():
            if step_name not in registry:
                logger.warning(
                    f"Unbekannter Deviation-Step '{step_name}' in Config, "
                    f"übersprungen. Verfügbar: {sorted(registry.keys())}"
                )
                continue

            params: Dict[str, Any] = dict(step_params or {})
            step_cls = registry[step_name]
            steps.append(step_cls(**params))

        return cls(steps=steps, tolerance_mm=tolerance_mm)