"""
Registrierungs-Sub-Pipeline für AP2.2.

Konkrete Steps:
- CoarsePCA  – Grob-Ausrichtung via PCA + Vorzeichen-Ambiguität
- (folgt)    – ICPFine
"""
from __future__ import annotations

from .coarse_pca import CoarsePCA
from .pipeline import RegistrationPipeline

__all__ = [
    "RegistrationPipeline",
    "CoarsePCA",
]