"""
VoxelDeviation – räumliche Aufschlüsselung der Distanzen in ein Voxel-Grid.

Gruppiert die signierten Distanzen aus dem vorherigen PointDistance-Step
in regelmäßige 3D-Zellen und aggregiert pro Voxel. Ergebnis: eine
ortsaufgelöste Karte der Abweichungen über das gesamte Bauteil.

Voraussetzung:
    PointDistance muss vor diesem Step laufen und data.distances_signed
    befüllen. Wenn kein PointDistance-Output vorhanden ist, wird der Step
    übersprungen.

Konfigurierbar über pipeline.yaml:
    voxel_size_mm:     Kantenlänge einer Voxel-Zelle in mm.
    min_points_per_voxel: Voxel mit weniger Punkten werden verworfen
                          (statistisch nicht aussagekräftig).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from ..base import DeviationStep
from ..reports import DeviationData

logger = logging.getLogger(__name__)


class VoxelDeviation(DeviationStep):
    """Räumliche Aggregation der signierten Distanzen in Voxel-Zellen."""

    def __init__(
        self,
        voxel_size_mm: float = 5.0,
        min_points_per_voxel: int = 3,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self.voxel_size_mm = float(voxel_size_mm)
        self.min_points_per_voxel = int(min_points_per_voxel)

    @property
    def name(self) -> str:
        return "voxel_deviation"

    def get_params(self) -> Dict[str, Any]:
        return {
            "voxel_size_mm": self.voxel_size_mm,
            "min_points_per_voxel": self.min_points_per_voxel,
        }

    def _apply(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        data: DeviationData,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if data.distances_signed is None or len(data.distances_signed) == 0:
            logger.warning(
                "  VoxelDeviation: keine distances_signed in DeviationData – "
                "PointDistance muss vorher laufen. Step übersprungen."
            )
            return {"skipped": True}

        source_pts = np.asarray(source.points)
        distances = np.asarray(data.distances_signed)

        if len(source_pts) != len(distances):
            raise ValueError(
                f"Punktzahl ({len(source_pts)}) und Distanz-Anzahl "
                f"({len(distances)}) inkonsistent."
            )

        # Voxel-Indizes: (i, j, k) pro Punkt
        min_corner = source_pts.min(axis=0)
        voxel_idx = np.floor(
            (source_pts - min_corner) / self.voxel_size_mm
        ).astype(np.int64)

        # Gruppierung über einen Hash (kombinierter Voxel-Key)
        # Kompaktere Repräsentation als tuples: pack in einen 64-bit int
        i, j, k = voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]
        # Kollisionsfreier Hash bei realistischen Bauteilgrößen (bis 2^20 Voxel je Achse)
        voxel_key = (i.astype(np.int64) << 40) | (j.astype(np.int64) << 20) | k.astype(np.int64)

        unique_keys, inverse_idx = np.unique(voxel_key, return_inverse=True)
        n_voxels_total = len(unique_keys)

        # Pro Voxel: Anzahl, signed mean, abs mean, rms
        counts = np.bincount(inverse_idx)
        sum_signed = np.bincount(inverse_idx, weights=distances)
        sum_abs = np.bincount(inverse_idx, weights=np.abs(distances))
        sum_sq = np.bincount(inverse_idx, weights=distances ** 2)

        # Voxel-Zentren rekonstruieren (Mittelwert der Punkt-Positionen pro Voxel)
        center_x = np.bincount(inverse_idx, weights=source_pts[:, 0]) / counts
        center_y = np.bincount(inverse_idx, weights=source_pts[:, 1]) / counts
        center_z = np.bincount(inverse_idx, weights=source_pts[:, 2]) / counts

        # Filter: min_points_per_voxel
        valid = counts >= self.min_points_per_voxel
        n_valid = int(valid.sum())

        centers = np.stack([center_x[valid], center_y[valid], center_z[valid]], axis=1)
        counts_v = counts[valid].astype(np.int64)
        mean_signed = sum_signed[valid] / counts_v
        mean_abs = sum_abs[valid] / counts_v
        rms = np.sqrt(sum_sq[valid] / counts_v)
        in_tol_rate = np.array([
            (np.abs(distances[inverse_idx == k]) <= data.tolerance_mm).mean()
            for k in np.where(valid)[0]
        ])

        # In DeviationData ablegen
        data.voxel_deviation = {
            "voxel_size_mm": self.voxel_size_mm,
            "centers": centers,          # (N_valid, 3)
            "counts": counts_v,          # (N_valid,)
            "mean_signed": mean_signed,  # (N_valid,)
            "mean_abs": mean_abs,        # (N_valid,)
            "rms": rms,                  # (N_valid,)
            "in_tolerance_rate": in_tol_rate,  # (N_valid,)
        }

        n_out_of_tol = int((mean_abs > data.tolerance_mm).sum())
        logger.info(
            f"  VoxelDeviation: {n_valid}/{n_voxels_total} Voxel gültig "
            f"(≥{self.min_points_per_voxel} Pkt/Voxel), "
            f"{n_out_of_tol} über Toleranz ({data.tolerance_mm:.2f}mm)"
        )

        return {
            "voxel_size_mm": self.voxel_size_mm,
            "n_voxels_total": n_voxels_total,
            "n_voxels_valid": n_valid,
            "n_voxels_out_of_tolerance": n_out_of_tol,
            "mean_of_voxel_means_abs_mm": float(mean_abs.mean()) if n_valid > 0 else 0.0,
            "max_voxel_mean_abs_mm": float(mean_abs.max()) if n_valid > 0 else 0.0,
        }