"""
PointDistance – signierte Punkt-zu-CAD-Distanz (AP2.2 Phase 1).

Methodik:
    Für jeden Scan-Punkt s wird der nächste CAD-Punkt c gesucht (KD-Tree).
    Die signierte Distanz ergibt sich aus dem Skalarprodukt mit der CAD-
    Normale am Punkt c:

        d_signed = (s - c) · n_c

    Vorzeichen-Konvention (mit nach außen zeigenden CAD-Normalen):
        d > 0  →  Scan-Punkt liegt außerhalb des idealen Volumens
                  (Material überschüssig, z.B. Grat, Schweißraupe)
        d < 0  →  Scan-Punkt liegt innerhalb (Material fehlt, z.B.
                  Wurzeldurchhang, Einbrand)
        d ≈ 0  →  Scan-Punkt liegt auf der CAD-Oberfläche

Diese Variante ist genauer als der reine euklidische Abstand zum
nächsten CAD-Punkt, weil sie nicht vom CAD-Sampling-Raster limitiert
ist: liegt ein Scan-Punkt direkt über einer Ebene, geht die Distanz
zur Ebene gegen Null, auch wenn der nächste diskrete CAD-Punkt seitlich
versetzt ist.

Aggregation:
    - Global: n_valid, mean_signed, mean_abs, rms, max_abs, p95, in_tolerance_rate
    - Per Label: dieselben Aggregate für jedes Segmentierungs-Label

Punkte mit |d| > max_distance werden als Ausreißer behandelt und nicht
in die Aggregate aufgenommen (z.B. Sub-Gap-Artefakte ohne CAD-Pendant).
Die rohen signierten Distanzen bleiben aber vollständig erhalten in
DeviationData.distances_signed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from ..base import DeviationStep
from ..reports import DeviationData

logger = logging.getLogger(__name__)


class PointDistance(DeviationStep):
    """Signierte Punkt-zu-CAD-Distanz Scan → CAD.

    Args:
        max_distance: Punkte mit |d| > max_distance werden als Ausreißer
            behandelt und aus den Aggregaten ausgeschlossen. mm.
        enabled: Step-Aktivierung.
    """

    def __init__(
        self,
        max_distance: float = 5.0,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self.max_distance = float(max_distance)

    @property
    def name(self) -> str:
        return "point_distance"

    def get_params(self) -> Dict[str, Any]:
        return {
            "max_distance": self.max_distance,
        }

    # ── Hauptlogik ────────────────────────────────────────────────────

    def _apply(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        data: DeviationData,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if not target.has_normals():
            raise ValueError(
                "PointDistance benötigt Normalen am Target. "
                "CAD-Wolken aus der API liefern diese mit."
            )

        source_pts = np.asarray(source.points)
        target_pts = np.asarray(target.points)
        target_normals = np.asarray(target.normals)
        n = len(source_pts)

        if n == 0:
            logger.warning("  PointDistance: leere Source – Step übersprungen.")
            return {"skipped": True, "n_points": 0}

        # Nearest-Neighbor-Suche via scipy (vektorisiert, schnell)
        try:
            from scipy.spatial import cKDTree
        except ImportError as e:
            raise ImportError(
                "PointDistance benötigt scipy. Installation: "
                "`uv add scipy` oder `pip install scipy`."
            ) from e

        tree = cKDTree(target_pts)
        _, nn_idx = tree.query(source_pts, k=1)

        c = target_pts[nn_idx]
        n_c = target_normals[nn_idx]

        # Signierte Distanz = (s - c) · n_c
        signed_distances = np.einsum("ij,ij->i", source_pts - c, n_c)
        abs_distances = np.abs(signed_distances)

        # Cutoff-Maske: was geht in die Aggregate?
        valid_mask = abs_distances <= self.max_distance
        in_tolerance = abs_distances <= data.tolerance_mm

        n_valid = int(valid_mask.sum())
        n_outlier = n - n_valid

        # ── Aggregate berechnen ──────────────────────────────────────
        global_agg = _aggregate(signed_distances, valid_mask, in_tolerance)

        per_region: Dict[int, Dict[str, float]] = {}
        if source_labels is not None:
            for label in np.unique(source_labels):
                label_mask = source_labels == label
                if not label_mask.any():
                    continue
                per_region[int(label)] = _aggregate(
                    signed_distances,
                    valid_mask & label_mask,
                    in_tolerance & label_mask,
                    n_points_label=int(label_mask.sum()),
                )

        # ── Ergebnisse in DeviationData ──────────────────────────────
        data.distances_signed = signed_distances
        data.in_tolerance = in_tolerance
        # Wenn schon Region-Metriken vorhanden (von anderem Step) – mergen
        if data.per_region_metrics:
            for k, v in per_region.items():
                data.per_region_metrics.setdefault(k, {}).update(v)
        else:
            data.per_region_metrics = per_region

        logger.info(
            f"  PointDistance: n={n:,}, valid={n_valid:,} "
            f"(outlier={n_outlier:,}, |d|>{self.max_distance:.1f}mm), "
            f"in_tol={in_tolerance.mean()*100:.1f}%, "
            f"|d|_mean={global_agg['mean_abs']:.3f}mm"
        )

        return {
            "n_points": n,
            "n_valid": n_valid,
            "n_outliers": n_outlier,
            "max_distance_cutoff_mm": self.max_distance,
            **{f"global_{k}": v for k, v in global_agg.items()},
            "n_regions": len(per_region),
        }


# ── Helpers ────────────────────────────────────────────────────────────

def _aggregate(
    signed_distances: np.ndarray,
    valid_mask: np.ndarray,
    in_tolerance_mask: np.ndarray,
    n_points_label: Optional[int] = None,
) -> Dict[str, float]:
    """Berechnet Aggregate über die Subset-Maske."""
    d_valid = signed_distances[valid_mask]
    abs_d_valid = np.abs(d_valid)
    n_total = int(n_points_label) if n_points_label is not None else len(signed_distances)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return {
            "n_points": n_total,
            "n_valid": 0,
            "mean_signed": float("nan"),
            "mean_abs": float("nan"),
            "rms": float("nan"),
            "max_abs": float("nan"),
            "p95": float("nan"),
            "in_tolerance_rate": 0.0,
        }

    return {
        "n_points": n_total,
        "n_valid": n_valid,
        "mean_signed": float(d_valid.mean()),
        "mean_abs": float(abs_d_valid.mean()),
        "rms": float(np.sqrt((d_valid ** 2).mean())),
        "max_abs": float(abs_d_valid.max()),
        "p95": float(np.percentile(abs_d_valid, 95)),
        "in_tolerance_rate": float(in_tolerance_mask.sum() / max(n_total, 1)),
    }