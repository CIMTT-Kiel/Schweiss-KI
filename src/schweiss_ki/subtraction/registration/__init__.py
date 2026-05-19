"""
Registrierungs-Sub-Pipeline für AP2.2.

Konkrete Steps:
- CoarsePCA  – Grob-Ausrichtung via PCA + Vorzeichen-Ambiguität
- ICPFine    – Fein-Registrierung via Point-to-Plane ICP
"""
from __future__ import annotations

from .coarse_pca import CoarsePCA
from .icp_fine import ICPFine
from .pipeline import RegistrationPipeline

__all__ = [
    "RegistrationPipeline",
    "CoarsePCA",
    "ICPFine",
]