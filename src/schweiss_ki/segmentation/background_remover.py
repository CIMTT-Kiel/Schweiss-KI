"""
Background-Entfernung via RANSAC-Ebenenfit (AP2.1 Phase 3).

Fittet die dominante horizontale Ebene (Werkstück-Oberseite) und markiert
alle UNLABELED-Punkte innerhalb der Toleranz als background (Label 0).
"""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from .base import SegmentationStep
from .labels import NAME_TO_ID, UNLABELED

logger = logging.getLogger(__name__)


class BackgroundRemover(SegmentationStep):
    """
    Klassifiziert die Werkstück-Oberseite als Background via RANSAC.

    Ablauf:
      1. Kandidaten-Filter: nur UNLABELED-Punkte, deren Normale grob parallel
         zu expected_normal liegt. Vorzeichen wird ignoriert, da "horizontal"
         beide Orientierungen umfasst (|cos| > threshold).
      2. RANSAC-Ebenenfit auf die Kandidaten → liefert Ebenenmodell.
      3. ALLE UNLABELED-Punkte innerhalb ransac_threshold der gefundenen
         Ebene werden als background markiert – auch solche, deren Normale
         nicht im Kandidaten-Set war (robuster gegen verrauschte Normalen).

    Voraussetzungen:
      - pcd hat Normalen (NormalEstimator im Preprocessing).

    Artefakte (für SegmentationReport):
      - plane_model:    [a, b, c, d] mit ax+by+cz+d=0, [a,b,c] normalisiert
      - plane_normal:   [a, b, c] als Unit-Vector
      - tilt_angle_deg: Winkel der gefundenen Ebene zur expected_normal
      - z_center:       Mittleres Z der Inlier (für downstream-Steps)
      - n_candidates:   Punkte nach Normalen-Vorfilter
      - n_inliers:      Punkte innerhalb ransac_threshold der gefundenen Ebene
    """

    def __init__(
        self,
        ransac_threshold: float = 0.25,
        max_iterations: int = 1000,
        ransac_n: int = 3,
        expected_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        normal_cos_threshold: float = 0.95,
        enabled: bool = True,
    ):
        """
        Args:
            ransac_threshold:     Max. Punkt→Ebene-Abstand für Inlier in mm.
                                  Default 0.25mm = Toleranzanforderung AP2.
            max_iterations:       RANSAC-Iterationen.
            ransac_n:             Minimale Punkte pro RANSAC-Hypothese (3 = Ebene).
            expected_normal:      Erwartete Ebenen-Normale (wird normalisiert).
                                  Default [0,0,1] = horizontale Werkstück-Oberseite.
            normal_cos_threshold: Cos-Schwelle für Kandidaten-Vorfilter.
                                  0.95 ≈ ±18°, 0.966 ≈ ±15°.
            enabled:              Step aktiv/inaktiv.
        """
        self._ransac_threshold = ransac_threshold
        self._max_iterations = max_iterations
        self._ransac_n = ransac_n

        exp_n = np.asarray(expected_normal, dtype=float)
        exp_n_len = np.linalg.norm(exp_n)
        if exp_n_len == 0.0:
            raise ValueError("expected_normal darf nicht der Nullvektor sein.")
        self._expected_normal = exp_n / exp_n_len

        self._normal_cos_threshold = normal_cos_threshold
        self._enabled = enabled
        self._last_artifacts: dict = {}

    @property
    def name(self) -> str:
        return "background_remover"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
    ) -> np.ndarray:
        if not pcd.has_normals():
            raise ValueError(
                f"{self.name} benötigt Normalen. "
                f"NormalEstimator im Preprocessing aktivieren."
            )

        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        labels_out = labels.copy()

        unlabeled_idx = np.where(labels == UNLABELED)[0]
        if len(unlabeled_idx) == 0:
            logger.warning(f"{self.name}: keine UNLABELED-Punkte, Step übersprungen.")
            self._last_artifacts = {}
            return labels_out

        # 1. Kandidaten-Filter: Normale grob parallel zu expected_normal
        cos_sim = np.abs(normals[unlabeled_idx] @ self._expected_normal)
        candidate_mask = cos_sim > self._normal_cos_threshold
        candidate_idx = unlabeled_idx[candidate_mask]

        if len(candidate_idx) < self._ransac_n:
            logger.warning(
                f"{self.name}: nur {len(candidate_idx)} Kandidaten nach Normalen-"
                f"Filter (threshold={self._normal_cos_threshold}), "
                f"RANSAC braucht mindestens {self._ransac_n}. Step übersprungen."
            )
            self._last_artifacts = {}
            return labels_out

        # 2. RANSAC-Ebenenfit auf Kandidaten
        candidate_pcd = pcd.select_by_index(candidate_idx.tolist())
        plane_model, _ = candidate_pcd.segment_plane(
            distance_threshold=self._ransac_threshold,
            ransac_n=self._ransac_n,
            num_iterations=self._max_iterations,
        )
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        normal_len = np.linalg.norm(plane_normal)
        normal_unit = plane_normal / normal_len
        d_unit = d / normal_len

        # 3. Alle UNLABELED-Punkte nahe der Ebene als background markieren
        distances = np.abs(points[unlabeled_idx] @ normal_unit + d_unit)
        inlier_mask = distances < self._ransac_threshold
        background_idx = unlabeled_idx[inlier_mask]

        labels_out[background_idx] = NAME_TO_ID["background"]

        # Artefakte für Report
        cos_to_expected = abs(normal_unit @ self._expected_normal)
        tilt_deg = float(np.degrees(np.arccos(np.clip(cos_to_expected, -1.0, 1.0))))
        z_center = float(np.mean(points[background_idx, 2])) if len(background_idx) > 0 else float("nan")

        self._last_artifacts = {
            "plane_model": [float(x) for x in (normal_unit[0], normal_unit[1], normal_unit[2], d_unit)],
            "plane_normal": normal_unit.tolist(),
            "tilt_angle_deg": tilt_deg,
            "z_center": z_center,
            "n_candidates": int(len(candidate_idx)),
            "n_inliers": int(len(background_idx)),
        }

        return labels_out

    def get_params(self) -> dict:
        return {
            "ransac_threshold": self._ransac_threshold,
            "max_iterations": self._max_iterations,
            "ransac_n": self._ransac_n,
            "expected_normal": self._expected_normal.tolist(),
            "normal_cos_threshold": self._normal_cos_threshold,
        }

    def get_artifacts(self) -> dict:
        return dict(self._last_artifacts)