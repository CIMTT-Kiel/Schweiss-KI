"""
Subtraction-Modul – AP2.2: Vergleich realer Scans mit CAD-Idealen.

Aufbau:
  registration/  – 3D-Registrierung (Scan ↔ CAD ausrichten)
  deviation/     – Differenzbildanalyse (Distanzen, Toleranzklassifikation)

Nutzungsfluss:
  1. RegistrationPipeline richtet source (Scan) an target (CAD) aus
  2. DeviationPipeline berechnet Abweichungen auf der ausgerichteten Wolke
  3. SubtractionReport bündelt beide Ergebnisse für das WeldVolumeModel
"""
from __future__ import annotations

from .base import DeviationStep, RegistrationStep
from .reports import (
    DeviationData,
    DeviationStepReport,
    RegistrationReport,
    RegistrationStepReport,
    SubtractionReport,
)
from .registration.pipeline import RegistrationPipeline
from .deviation.pipeline import DeviationPipeline


__all__ = [
    # Base interfaces
    "RegistrationStep",
    "DeviationStep",
    # Reports
    "RegistrationStepReport",
    "RegistrationReport",
    "DeviationStepReport",
    "DeviationData",
    "SubtractionReport",
    # Pipelines
    "RegistrationPipeline",
    "DeviationPipeline",
]