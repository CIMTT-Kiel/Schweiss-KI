"""
Differenzanalyse-Sub-Pipeline für AP2.2 und AP2.3.

Konkrete Steps:
- PointDistance          – Signierte Punkt-zu-CAD-Distanz (global + per Label)
- VoxelDeviation         – Räumliche Aufschlüsselung in Voxel-Zellen
- ComponentRegistration  – Werkstückweise Registrierung → relative Lage (6 DOF)
- GapProfile             – Wurzelspalt-Profil entlang der Naht (V-Naht-spezifisch)
"""
from __future__ import annotations

from .component_registration import ComponentRegistration
from .gap_profile import GapProfile
from .pipeline import DeviationPipeline
from .point_distance import PointDistance
from .voxel_deviation import VoxelDeviation

__all__ = [
    "DeviationPipeline",
    "ComponentRegistration",
    "GapProfile",
    "PointDistance",
    "VoxelDeviation",
]