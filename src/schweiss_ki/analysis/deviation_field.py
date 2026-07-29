"""Per-Punkt-Abweichungsfeld für Visualisierung und Dashboard.

Die Batch-Reports speichern nur Aggregate (Regionen, Voxel), nicht den
signierten Abstand je Punkt. Dieses Modul rechnet ihn aus der gespeicherten,
bereits ausgerichteten Scanwolke gegen das CAD-Ideal nach — dieselbe Rechnung
wie `subtraction/deviation/point_distance.py`:

    d = (s − c) · n_c      c = nächster CAD-Punkt, n_c dessen Normale

    d > 0  Material steht über (Grat, Raupe)
    d < 0  Material fehlt (Durchhang, Einbrand)
    |d| ≤ Toleranz  in Ordnung

Ergebnis wird je Fall nach `signed_distance.npy` zwischengespeichert, damit
statische Bilder und interaktives Dashboard dieselbe Zahlengrundlage teilen und
die KD-Baum-Suche nicht doppelt läuft.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

DEFAULT_CAD_STEM = "Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt"
TOLERANCE_MM = 0.25
OUTLIER_CUTOFF_MM = 5.0


class CadReference(NamedTuple):
    """CAD-Ideal mit vorbereitetem KD-Baum, einmal laden und wiederverwenden."""
    points: np.ndarray
    normals: np.ndarray
    tree: cKDTree

    @property
    def top_mask(self) -> np.ndarray:
        """Nur die Oberseite (Normale zeigt nach oben) — für die Soll-Ansicht."""
        return self.normals[:, 2] > 0.5


def load_cad_reference(outputs_dir: Path, cad_stem: str = DEFAULT_CAD_STEM) -> CadReference:
    cad_path = outputs_dir / "cad" / cad_stem / "pointcloud.ply"
    cad = o3d.io.read_point_cloud(str(cad_path))
    if not cad.has_normals():
        raise ValueError(f"CAD-Wolke ohne Normalen: {cad_path}")
    pts = np.asarray(cad.points)
    return CadReference(points=pts, normals=np.asarray(cad.normals), tree=cKDTree(pts))


class CaseField(NamedTuple):
    points: np.ndarray          # (n, 3) ausgerichtete Scanpunkte
    labels: np.ndarray          # (n,) Segmentierungslabel 0/1/2
    signed: np.ndarray          # (n,) signierter Abstand zum CAD, mm
    in_tolerance: np.ndarray    # (n,) bool, |signed| ≤ Toleranz

    @property
    def in_tolerance_rate(self) -> float:
        return float(self.in_tolerance.mean())


def signed_distance_field(
    case_dir: Path,
    cad: CadReference,
    tolerance_mm: float = TOLERANCE_MM,
    use_cache: bool = True,
) -> CaseField:
    """Signiertes Abweichungsfeld eines Falls, mit npy-Cache."""
    scan = o3d.io.read_point_cloud(str(case_dir / "pointcloud.ply"))
    pts = np.asarray(scan.points)
    labels = np.load(case_dir / "labels.npy")

    cache = case_dir / "signed_distance.npy"
    if use_cache and cache.exists() and cache.stat().st_mtime >= (case_dir / "pointcloud.ply").stat().st_mtime:
        signed = np.load(cache)
        if signed.shape[0] != pts.shape[0]:          # Wolke neu, Cache veraltet
            signed = _compute(pts, cad)
            np.save(cache, signed)
    else:
        signed = _compute(pts, cad)
        if use_cache:
            np.save(cache, signed)

    return CaseField(points=pts, labels=labels, signed=signed,
                     in_tolerance=np.abs(signed) <= tolerance_mm)


def _compute(scan_pts: np.ndarray, cad: CadReference) -> np.ndarray:
    _, idx = cad.tree.query(scan_pts, k=1)
    return np.einsum("ij,ij->i", scan_pts - cad.points[idx], cad.normals[idx])


def subsample(n: int, target: int, seed: int = 0) -> np.ndarray:
    """Reproduzierbare Index-Teilmenge für die Darstellung dichter Wolken."""
    if n <= target:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=target, replace=False)
