"""
Segmentierungs-Pipeline für AP2.1 (Phase 3).

Klassifiziert vorverarbeitete Punktwolken in:
    0 = background
    1 = flank_a           (linke V-Fase,  n_x > 0)
    2 = flank_b           (rechte V-Fase, n_x < 0)
    3 = gap_region
    4 = sub_gap_artifacts
"""
from .background_remover import BackgroundRemover
from .base import (
    SegmentationPipeline,
    SegmentationReport,
    SegmentationStep,
    SegmentationStepReport,
)
from .flank_segmenter import FlankSegmenter
from .gap_classifier import GapClassifier
from .labels import LABEL_DESCRIPTIONS, LABELS, NAME_TO_ID, UNLABELED

__all__ = [
    # Pipeline
    "SegmentationPipeline",
    "SegmentationReport",
    "SegmentationStep",
    "SegmentationStepReport",
    # Steps
    "BackgroundRemover",
    "FlankSegmenter",
    "GapClassifier",
    # Labels
    "LABELS",
    "LABEL_DESCRIPTIONS",
    "NAME_TO_ID",
    "UNLABELED",
]