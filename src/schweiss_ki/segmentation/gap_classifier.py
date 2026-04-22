"""
Gap-Region-Klassifikator (AP2.1 Phase 3).

Klassifiziert UNLABELED-Punkte als gap_region oder sub_gap_artifacts
basierend auf den X/Z-Grenzen der zuvor segmentierten Flanken.

Sub-Spalt-Artefakte (Punkte unterhalb der Flanken-Unterkante, z.B. durch
CMM-Durchstich bei durchgehendem Spalt) werden als eigenes Label geführt,
damit sie später nachbearbeitet werden können.
"""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from .base import SegmentationStep
from .labels import NAME_TO_ID, UNLABELED

logger = logging.getLogger(__name__)


class GapClassifier(SegmentationStep):
    """
    Klassifiziert den Spaltbereich zwischen den Flanken.

    Ablauf:
      1. X/Z-Grenzen aus bereits segmentierten Flanken (Label 1 + 2) berechnen:
         - x_min / x_max: Extremwerte aller Flanken-Punkte (+ x_margin)
         - z_lower:       Robuster Min-Z (z_lower_quantile der Flanken-Z-Werte)
         - z_upper:       Max-Z der Flanken (nur als Metrik, nicht als Filter)
      2. UNLABELED-Punkte im X-Bereich klassifizieren:
         - Z >= z_lower → gap_region (Label 3)
         - Z <  z_lower → sub_gap_artifacts (Label 4) oder background (Label 0),
                          je nach separate_sub_gap_artifacts.
      3. UNLABELED-Punkte außerhalb X-Bereich bleiben UNLABELED
         (werden am Pipeline-Ende zu background konvertiert).

    Voraussetzung:
        labels enthält flank_a (=1) und flank_b (=2), z.B. nach FlankSegmenter.

    Artefakte:
        x_min, x_max, z_lower, z_upper, n_gap, n_sub_gap, gap_width_by_y
    """

    def __init__(
        self,
        z_lower_quantile: float = 0.05,
        x_margin: float = 0.5,
        separate_sub_gap_artifacts: bool = True,
        gap_width_y_bins: int = 20,
        enabled: bool = True,
    ):
        """
        Args:
            z_lower_quantile:           Quantil der Flanken-Z für robusten
                                        Min-Z (robuster als min()). 0.05 = P5.
            x_margin:                   Zusätzlicher Puffer in mm auf die
                                        X-Bounds der Flanken. Fängt Grenzpunkte
                                        am Übergang Fase → Gap ab.
            separate_sub_gap_artifacts: True → Artefakte unter Flanken-Z als
                                        Label 4. False → als Label 0 (Background).
            gap_width_y_bins:           Anzahl Y-Slices für die Spaltbreiten-
                                        Metrik (gap_width_by_y).
            enabled:                    Step aktiv/inaktiv.
        """
        self._z_lower_quantile = z_lower_quantile
        self._x_margin = x_margin
        self._separate_sub_gap_artifacts = separate_sub_gap_artifacts
        self._gap_width_y_bins = gap_width_y_bins
        self._enabled = enabled
        self._last_artifacts: dict = {}

    @property
    def name(self) -> str:
        return "gap_classifier"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(pcd.points)
        labels_out = labels.copy()

        flank_a_id = NAME_TO_ID["flank_a"]
        flank_b_id = NAME_TO_ID["flank_b"]
        flank_mask = (labels_out == flank_a_id) | (labels_out == flank_b_id)

        if int(flank_mask.sum()) < 10:
            logger.warning(
                f"{self.name}: zu wenige Flanken-Punkte "
                f"({int(flank_mask.sum())}). Step übersprungen."
            )
            self._last_artifacts = {}
            return labels_out

        flank_points = points[flank_mask]
        x_min = float(flank_points[:, 0].min()) - self._x_margin
        x_max = float(flank_points[:, 0].max()) + self._x_margin
        z_lower = float(np.quantile(flank_points[:, 2], self._z_lower_quantile))
        z_upper = float(flank_points[:, 2].max())

        unlabeled_idx = np.where(labels_out == UNLABELED)[0]
        unlabeled_pts = points[unlabeled_idx]

        in_x = (unlabeled_pts[:, 0] >= x_min) & (unlabeled_pts[:, 0] <= x_max)
        at_or_above_lower = unlabeled_pts[:, 2] >= z_lower

        gap_mask = in_x & at_or_above_lower
        sub_gap_mask = in_x & ~at_or_above_lower

        labels_out[unlabeled_idx[gap_mask]] = NAME_TO_ID["gap_region"]

        sub_gap_target = (
            NAME_TO_ID["sub_gap_artifacts"]
            if self._separate_sub_gap_artifacts
            else NAME_TO_ID["background"]
        )
        labels_out[unlabeled_idx[sub_gap_mask]] = sub_gap_target

        # Spaltbreite entlang Y (Validierung: erwartet 1.0→2.5mm linear)
        gap_width_by_y = self._compute_gap_width_by_y(points, labels_out)

        self._last_artifacts = {
            "x_min": x_min,
            "x_max": x_max,
            "z_lower": z_lower,
            "z_upper": z_upper,
            "n_gap": int(gap_mask.sum()),
            "n_sub_gap": int(sub_gap_mask.sum()),
            "gap_width_by_y": gap_width_by_y.tolist() if gap_width_by_y.size else [],
        }

        return labels_out

    def _compute_gap_width_by_y(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Spaltbreite pro Y-Slice = minimaler X-Abstand zwischen Flanke A und B
        im Slice (= Rand von A bis Rand von B am Grund).

        Returns:
            Array shape (N, 2) mit Spalten [y_center, width]. Leere Slices
            (zu wenige Punkte) werden übersprungen.
        """
        mask_a = labels == NAME_TO_ID["flank_a"]
        mask_b = labels == NAME_TO_ID["flank_b"]
        if not (mask_a.any() and mask_b.any()):
            return np.zeros((0, 2))

        pts_a = points[mask_a]
        pts_b = points[mask_b]
        y_min = max(float(pts_a[:, 1].min()), float(pts_b[:, 1].min()))
        y_max = min(float(pts_a[:, 1].max()), float(pts_b[:, 1].max()))
        if y_max <= y_min:
            return np.zeros((0, 2))

        y_edges = np.linspace(y_min, y_max, self._gap_width_y_bins + 1)
        widths: list[list[float]] = []
        for i in range(self._gap_width_y_bins):
            in_a = (pts_a[:, 1] >= y_edges[i]) & (pts_a[:, 1] < y_edges[i + 1])
            in_b = (pts_b[:, 1] >= y_edges[i]) & (pts_b[:, 1] < y_edges[i + 1])
            if int(in_a.sum()) < 3 or int(in_b.sum()) < 3:
                continue
            # Minimaler X-Abstand: rechter Rand von A, linker Rand von B
            x_a_right = float(pts_a[in_a, 0].max())
            x_b_left = float(pts_b[in_b, 0].min())
            width = x_b_left - x_a_right
            y_center = float((y_edges[i] + y_edges[i + 1]) / 2)
            widths.append([y_center, width])

        return np.array(widths) if widths else np.zeros((0, 2))

    def get_params(self) -> dict:
        return {
            "z_lower_quantile": self._z_lower_quantile,
            "x_margin": self._x_margin,
            "separate_sub_gap_artifacts": self._separate_sub_gap_artifacts,
            "gap_width_y_bins": self._gap_width_y_bins,
        }

    def get_artifacts(self) -> dict:
        return dict(self._last_artifacts)