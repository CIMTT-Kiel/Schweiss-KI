"""
Preprocessing-Modul für Schweiß-KI AP2.1 Phase 2.

Konfigurierbare Pipeline zur Aufbereitung von Punktwolken.

Schnelleinstieg:
    from schweiss_ki.preprocessing import (
        PreprocessingPipeline,
        StatisticalOutlierFilter,
        VoxelGridDownsampler,
        NormalEstimator,
    )

    pipeline = PreprocessingPipeline([
        StatisticalOutlierFilter(nb_neighbors=20, std_ratio=2.0),
        VoxelGridDownsampler(voxel_size=0.5),
        NormalEstimator(radius=2.0),
    ])
    pcd_clean, report = pipeline.process(pcd_raw)
    print(report.summary())
"""

from .base import (
    PreprocessingReport,
    PreprocessingStep,
    StepReport,
)
from .pipeline import PreprocessingPipeline
from .downsampling import RandomDownsampler, VoxelGridDownsampler
from .filtering import RadiusOutlierFilter, StatisticalOutlierFilter
from .normalization import NormalEstimator

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    # Base
    "PreprocessingReport",
    "PreprocessingStep",
    "StepReport",
    # Filtering
    "StatisticalOutlierFilter",
    "RadiusOutlierFilter",
    # Downsampling
    "VoxelGridDownsampler",
    "RandomDownsampler",
    # Enrichment
    "NormalEstimator",
]