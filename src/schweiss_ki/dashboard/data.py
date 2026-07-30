"""Datenzugriff fürs Dashboard: Fallentdeckung und gecachtes Laden.

Alles wird live aus den Batch-Ausgaben gerechnet. Die teuren Schritte —
CAD-KD-Baum und die per-Punkt-Abstände — sind gecacht, damit der Fallwechsel
schnell bleibt. Nichts ist fest verdrahtet: `discover_cases` scannt das
Ausgabeverzeichnis, sodass neue (auch reale) Batches automatisch erscheinen.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from schweiss_ki.analysis.deviation_field import (
    CadReference, CaseField, load_cad_reference, signed_distance_field,
    subsample,
)

DEFAULT_OUTPUTS = Path(__file__).resolve().parents[3] / "data" / "outputs"


def discover_cases(outputs_dir: Path) -> list[str]:
    """Alle Fallordner mit gültigem Report — alphabetisch, ohne CAD-Cache."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return []
    return sorted(
        p.name for p in outputs_dir.iterdir()
        if p.is_dir() and p.name != "cad"
        and (p / "subtraction_report.json").exists())


@functools.lru_cache(maxsize=2)
def cad_reference(outputs_dir: str) -> CadReference:
    return load_cad_reference(Path(outputs_dir))


@functools.lru_cache(maxsize=1)
def _cad_top(outputs_dir: str) -> np.ndarray:
    cad = cad_reference(outputs_dir)
    return cad.points[cad.top_mask]


def cad_top_points(outputs_dir: Path, n: int) -> np.ndarray:
    """Oberseite des CAD-Ideals, für die Anzeige ausgedünnt."""
    top = _cad_top(str(outputs_dir))
    return top[subsample(len(top), n, seed=7)]


@functools.lru_cache(maxsize=16)
def _field_cached(outputs_dir: str, case: str) -> CaseField:
    cad = cad_reference(outputs_dir)
    return signed_distance_field(Path(outputs_dir) / case, cad)


def case_field(outputs_dir: Path, case: str) -> CaseField:
    """Signiertes Abweichungsfeld des Falls (mit npy- und In-Memory-Cache)."""
    return _field_cached(str(outputs_dir), case)


def downsampled_cloud(field: CaseField, n: int):
    """(Punkte, signierte Abstände) für die Anzeige. n=0 → volle Dichte."""
    if n <= 0 or n >= len(field.points):
        return field.points, field.signed
    idx = subsample(len(field.points), n, seed=3)
    return field.points[idx], field.signed[idx]


@functools.lru_cache(maxsize=64)
def load_report(outputs_dir: str, case: str) -> dict:
    with open(Path(outputs_dir) / case / "subtraction_report.json",
              encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=64)
def source_type(outputs_dir: str, case: str) -> str:
    """'synthetic' / 'real' / … aus metadata.json — treibt die Kennzeichnung."""
    p = Path(outputs_dir) / case / "metadata.json"
    if not p.exists():
        return "unbekannt"
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("source_type", "unbekannt")
    except (json.JSONDecodeError, OSError):
        return "unbekannt"
