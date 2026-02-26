"""
Preprocessing-Modul für Schweiß-KI AP2.1 Phase 2.

Konfigurierbare Pipeline zur Aufbereitung von Punktwolken.

Schnelleinstieg (manuell):
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

Schnelleinstieg (aus YAML):
    from schweiss_ki.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline.from_config(
        "configs/preprocessing.yaml", source_type="real"
    )
    pcd_clean, report = pipeline.process(pcd_raw)
"""

from .base import (
    PreprocessingPipeline,
    PreprocessingReport,
    PreprocessingStep,
    StepReport,
)
from .downsampling import RandomDownsampler, VoxelGridDownsampler
from .filtering import RadiusOutlierFilter, StatisticalOutlierFilter
from .normalization import Centerer, NormalEstimator

__all__ = [
    # Base
    "PreprocessingPipeline",
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
    "Centerer",
]