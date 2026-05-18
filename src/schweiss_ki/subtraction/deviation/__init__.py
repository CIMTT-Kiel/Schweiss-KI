"""
Differenzanalyse-Sub-Pipeline für AP2.2.

Konkrete Steps werden hier nach und nach hinzugefügt:
- PointDistance       – Bidirektionale signierte Distanz Scan ↔ CAD
- PerRegionMetrics    – Aggregate pro Segmentierungs-Label
- GapProfile          – Spaltmaß/Wurzeltiefe/Öffnungswinkel entlang Y
- ToleranceClassifier – ±0.25 mm Klassifikation pro Punkt und Region
"""
from __future__ import annotations

from .pipeline import DeviationPipeline

__all__ = [
    "DeviationPipeline",
]