"""
Gap-Region-Klassifikator (AP2.1 Phase 3).

Klassifiziert UNLABELED-Punkte als gap_region oder sub_gap_artifacts
basierend auf den Gap-/Tiefen-Grenzen der zuvor segmentierten Flanken.

Sub-Spalt-Artefakte (Punkte unterhalb der Flanken-Unterkante, z.B. durch
CMM-Durchstich bei durchgehendem Spalt) werden als eigenes Label geführt,
damit sie später nachbearbeitet werden können.

Achsen-Konvention (identisch zur Subtraktions-Stage):
    seam_axis     – Naht-Längsrichtung          (Default 0 = X)
    gap_axis      – Spalt-Querrichtung          (Default 1 = Y)
    vertical_axis – Tiefe                       (Default 2 = Z)
"""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from .base import SegmentationStep, validate_axes
from .labels import NAME_TO_ID, UNLABELED

logger = logging.getLogger(__name__)


class GapClassifier(SegmentationStep):
    """
    Klassifiziert den Spaltbereich zwischen den Flanken.

    Ablauf:
      1. Gap-/Tiefen-Grenzen aus bereits segmentierten Flanken (Label 1 + 2):
         - gap_min / gap_max: Extremwerte aller Flanken-Punkte entlang
                              gap_axis (± gap_margin)
         - z_lower:           Robustes Minimum entlang vertical_axis
                              (z_lower_quantile der Flanken-Tiefenwerte)
         - z_upper:           Maximum entlang vertical_axis (nur Metrik)
      2. UNLABELED-Punkte im gap_axis-Bereich klassifizieren:
         - Tiefe >= z_lower → gap_region (Label 3)
         - Tiefe <  z_lower → sub_gap_artifacts (Label 4) oder background (0),
                              je nach separate_sub_gap_artifacts.
      3. UNLABELED-Punkte außerhalb des gap_axis-Bereichs bleiben UNLABELED
         (werden am Pipeline-Ende zu background konvertiert).

    Voraussetzung:
        labels enthält flank_a (=1) und flank_b (=2), z.B. nach FlankSegmenter.

    Artefakte:
        gap_min, gap_max, z_lower, z_upper, n_gap, n_sub_gap, gap_width_by_seam
    """

    def __init__(
        self,
        z_lower_quantile: float = 0.05,
        gap_margin: float = 0.5,
        separate_sub_gap_artifacts: bool = True,
        gap_width_bins: int = 20,
        seam_axis: int = 0,
        gap_axis: int = 1,
        vertical_axis: int = 2,
        enabled: bool = True,
    ):
        """
        Args:
            z_lower_quantile:           Quantil der Flanken-Tiefenwerte für ein
                                        robustes Minimum (robuster als min()).
                                        0.05 = P5.
            gap_margin:                 Zusätzlicher Puffer in mm auf die
                                        gap_axis-Bounds der Flanken. Fängt
                                        Grenzpunkte am Übergang Fase → Gap ab.
            separate_sub_gap_artifacts: True → Artefakte unter Flanken-Tiefe als
                                        Label 4. False → als Label 0 (Background).
            gap_width_bins:             Anzahl Slices entlang seam_axis für die
                                        Spaltbreiten-Metrik (gap_width_by_seam).
            seam_axis:                  Naht-Längsrichtung (0=X, 1=Y, 2=Z).
            gap_axis:                   Spalt-Querrichtung.
            vertical_axis:              Tiefenrichtung.
            enabled:                    Step aktiv/inaktiv.
        """
        self._z_lower_quantile = z_lower_quantile
        self._gap_margin = gap_margin
        self._separate_sub_gap_artifacts = separate_sub_gap_artifacts
        self._gap_width_bins = gap_width_bins
        self._seam_axis, self._gap_axis, self._vertical_axis = validate_axes(
            seam_axis, gap_axis, vertical_axis
        )
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
        gap_min = float(flank_points[:, self._gap_axis].min()) - self._gap_margin
        gap_max = float(flank_points[:, self._gap_axis].max()) + self._gap_margin
        z_lower = float(
            np.quantile(flank_points[:, self._vertical_axis], self._z_lower_quantile)
        )
        z_upper = float(flank_points[:, self._vertical_axis].max())

        unlabeled_idx = np.where(labels_out == UNLABELED)[0]
        unlabeled_pts = points[unlabeled_idx]

        in_gap_span = (unlabeled_pts[:, self._gap_axis] >= gap_min) & (
            unlabeled_pts[:, self._gap_axis] <= gap_max
        )
        at_or_above_lower = unlabeled_pts[:, self._vertical_axis] >= z_lower

        gap_mask = in_gap_span & at_or_above_lower
        sub_gap_mask = in_gap_span & ~at_or_above_lower

        labels_out[unlabeled_idx[gap_mask]] = NAME_TO_ID["gap_region"]

        sub_gap_target = (
            NAME_TO_ID["sub_gap_artifacts"]
            if self._separate_sub_gap_artifacts
            else NAME_TO_ID["background"]
        )
        labels_out[unlabeled_idx[sub_gap_mask]] = sub_gap_target

        # Spaltbreite entlang der Naht (Validierung: erwartet 1.0→2.5mm linear)
        gap_width_by_seam = self._compute_gap_width_by_seam(points, labels_out)

        self._last_artifacts = {
            "gap_min": gap_min,
            "gap_max": gap_max,
            "z_lower": z_lower,
            "z_upper": z_upper,
            "n_gap": int(gap_mask.sum()),
            "n_sub_gap": int(sub_gap_mask.sum()),
            "gap_width_by_seam": (
                gap_width_by_seam.tolist() if gap_width_by_seam.size else []
            ),
        }

        return labels_out

    def _compute_gap_width_by_seam(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Spaltbreite pro Slice entlang seam_axis = minimaler gap_axis-Abstand
        zwischen Flanke A und B im Slice (= Rand von A bis Rand von B am Grund).

        Returns:
            Array shape (N, 2) mit Spalten [seam_center, width]. Leere Slices
            (zu wenige Punkte) werden übersprungen.
        """
        mask_a = labels == NAME_TO_ID["flank_a"]
        mask_b = labels == NAME_TO_ID["flank_b"]
        if not (mask_a.any() and mask_b.any()):
            return np.zeros((0, 2))

        pts_a = points[mask_a]
        pts_b = points[mask_b]
        seam_a = pts_a[:, self._seam_axis]
        seam_b = pts_b[:, self._seam_axis]
        seam_min = max(float(seam_a.min()), float(seam_b.min()))
        seam_max = min(float(seam_a.max()), float(seam_b.max()))
        if seam_max <= seam_min:
            return np.zeros((0, 2))

        seam_edges = np.linspace(seam_min, seam_max, self._gap_width_bins + 1)
        widths: list[list[float]] = []
        for i in range(self._gap_width_bins):
            in_a = (seam_a >= seam_edges[i]) & (seam_a < seam_edges[i + 1])
            in_b = (seam_b >= seam_edges[i]) & (seam_b < seam_edges[i + 1])
            if int(in_a.sum()) < 3 or int(in_b.sum()) < 3:
                continue
            # Minimaler gap_axis-Abstand: innerer Rand von A bis innerer Rand von B.
            # Flank A liegt per Konvention auf der negativen gap_axis-Seite.
            a_inner = float(pts_a[in_a, self._gap_axis].max())
            b_inner = float(pts_b[in_b, self._gap_axis].min())
            width = b_inner - a_inner
            seam_center = float((seam_edges[i] + seam_edges[i + 1]) / 2)
            widths.append([seam_center, width])

        return np.array(widths) if widths else np.zeros((0, 2))

    def get_params(self) -> dict:
        return {
            "z_lower_quantile": self._z_lower_quantile,
            "gap_margin": self._gap_margin,
            "separate_sub_gap_artifacts": self._separate_sub_gap_artifacts,
            "gap_width_bins": self._gap_width_bins,
            "seam_axis": self._seam_axis,
            "gap_axis": self._gap_axis,
            "vertical_axis": self._vertical_axis,
        }

    def get_artifacts(self) -> dict:
        return dict(self._last_artifacts)