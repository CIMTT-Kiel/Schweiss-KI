"""
Flanken-Segmentierung via zweimal RANSAC (AP2.1 Phase 3).

Segmentiert die beiden schrägen V-Fasen getrennt, mit Normalen-Vorfilter
zur Unterscheidung links (Flank A) vs. rechts (Flank B).

Konvention:
    Flank A = linke V-Fase,  Normale ≈ ( cos(α),  0, sin(α)),  n_x > 0
    Flank B = rechte V-Fase, Normale ≈ (-cos(α),  0, sin(α)),  n_x < 0
    α = expected_flank_angle_deg (Winkel der Flanke zur Vertikalen)

    Für die nominelle 60° V-Naht (Öffnungswinkel = 60°, je 30° pro Seite):
        Flank A expected: (0.866, 0, 0.5)
        Flank B expected: (-0.866, 0, 0.5)

Voraussetzung:
    Normalen müssen konsistent "nach außen" orientiert sein (d.h. vom
    Werkstück-Inneren weg). Bei CMM-Scan von oben entspricht das n_z > 0
    für die Flanken. Liefert das Preprocessing flipped Normalen, findet
    dieser Step 0 Kandidaten → Warnung im Report.
"""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from .base import SegmentationStep
from .labels import NAME_TO_ID, UNLABELED

logger = logging.getLogger(__name__)


class FlankSegmenter(SegmentationStep):
    """
    Segmentiert beide V-Fasen in einem Step via zweimal RANSAC.

    Ablauf pro Seite:
      1. Kandidaten-Filter: signed cos sim mit expected_normal > threshold.
         Vorzeichen matters – unterscheidet A (n_x > 0) von B (n_x < 0).
      2. RANSAC-Ebenenfit auf Kandidaten.
      3. RANSAC-Inlier (im Kandidaten-Set) → Label flank_a bzw. flank_b.

    Strenger Ansatz (im Gegensatz zu background_remover): nur Punkte, die
    den Normalen-Filter passiert haben UND RANSAC-Inlier sind, werden
    klassifiziert. Grund: Am V-Grund können Flanke A und B geometrisch
    nahe beieinander liegen; die Normal-Richtung ist das einzige
    verlässliche Unterscheidungsmerkmal.

    Artefakte:
        flank_a / flank_b jeweils mit:
            status, plane_model, plane_normal, angle_from_vertical_deg,
            n_candidates, n_inliers
    """

    def __init__(
        self,
        ransac_threshold: float = 0.25,
        max_iterations: int = 1000,
        ransac_n: int = 3,
        expected_flank_angle_deg: float = 30.0,
        normal_cos_threshold: float = 0.85,
        enabled: bool = True,
    ):
        """
        Args:
            ransac_threshold:         Max. Punkt→Ebene-Abstand für Inlier (mm).
                                      Default 0.25mm = Toleranzanforderung AP2.
            max_iterations:           RANSAC-Iterationen.
            ransac_n:                 Min. Punkte pro RANSAC-Hypothese.
            expected_flank_angle_deg: Erwartete Flankenneigung zur Vertikalen
                                      in Grad. 30° = nominelle 60° V-Naht.
            normal_cos_threshold:     Signed cos-Schwelle für Kandidaten-
                                      Vorfilter. 0.85 ≈ ±32°, 0.9 ≈ ±26°,
                                      0.95 ≈ ±18°.
            enabled:                  Step aktiv/inaktiv.
        """
        self._ransac_threshold = ransac_threshold
        self._max_iterations = max_iterations
        self._ransac_n = ransac_n
        self._expected_flank_angle_deg = expected_flank_angle_deg
        self._normal_cos_threshold = normal_cos_threshold
        self._enabled = enabled

        alpha = np.deg2rad(expected_flank_angle_deg)
        # Flank A (links): Normale nach rechts-oben
        self._expected_normal_a = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
        # Flank B (rechts): Normale nach links-oben
        self._expected_normal_b = np.array([-np.cos(alpha), 0.0, np.sin(alpha)])

        self._last_artifacts: dict = {}

    @property
    def name(self) -> str:
        return "flank_segmenter"

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

        labels_out = labels.copy()
        artifacts: dict = {}

        for side_key, expected_normal, target_label_name in [
            ("flank_a", self._expected_normal_a, "flank_a"),
            ("flank_b", self._expected_normal_b, "flank_b"),
        ]:
            labels_out, side_artifacts = self._segment_one_side(
                pcd, labels_out, expected_normal, target_label_name, side_key
            )
            artifacts[side_key] = side_artifacts

        self._last_artifacts = artifacts
        return labels_out

    def _segment_one_side(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
        expected_normal: np.ndarray,
        target_label_name: str,
        side_name: str,
    ) -> tuple[np.ndarray, dict]:
        normals = np.asarray(pcd.normals)

        unlabeled_idx = np.where(labels == UNLABELED)[0]
        if len(unlabeled_idx) == 0:
            return labels, {
                "status": "no_unlabeled_points",
                "n_candidates": 0,
                "n_inliers": 0,
            }

        # Signed cos similarity (Vorzeichen matters für Seitenunterscheidung)
        cos_sim = normals[unlabeled_idx] @ expected_normal
        candidate_mask = cos_sim > self._normal_cos_threshold
        candidate_idx = unlabeled_idx[candidate_mask]

        if len(candidate_idx) < self._ransac_n:
            logger.warning(
                f"{self.name} ({side_name}): nur {len(candidate_idx)} Kandidaten "
                f"nach Normalen-Filter (threshold={self._normal_cos_threshold}). "
                f"Seite übersprungen."
            )
            return labels, {
                "status": "insufficient_candidates",
                "n_candidates": int(len(candidate_idx)),
                "n_inliers": 0,
            }

        # RANSAC auf Kandidaten-Teilwolke
        candidate_pcd = pcd.select_by_index(candidate_idx.tolist())
        plane_model, inlier_local = candidate_pcd.segment_plane(
            distance_threshold=self._ransac_threshold,
            ransac_n=self._ransac_n,
            num_iterations=self._max_iterations,
        )
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        normal_len = np.linalg.norm(plane_normal)
        normal_unit = plane_normal / normal_len
        d_unit = d / normal_len

        # Plane normal ggf. flippen, damit Richtung mit expected_normal übereinstimmt
        # (segment_plane liefert beliebige Orientierung)
        if normal_unit @ expected_normal < 0:
            normal_unit = -normal_unit
            d_unit = -d_unit

        # Lokale Inlier-Indizes (innerhalb Kandidaten-Set) → Original-Indizes
        inlier_orig = candidate_idx[np.asarray(inlier_local, dtype=int)]

        labels_out = labels.copy()
        labels_out[inlier_orig] = NAME_TO_ID[target_label_name]

        # Winkel der gefitteten Ebene zur Vertikalen (arcsin(|n_z|))
        angle_from_vertical_deg = float(
            np.degrees(np.arcsin(min(abs(float(normal_unit[2])), 1.0)))
        )

        side_artifacts = {
            "status": "ok",
            "plane_model": [
                float(normal_unit[0]),
                float(normal_unit[1]),
                float(normal_unit[2]),
                float(d_unit),
            ],
            "plane_normal": normal_unit.tolist(),
            "angle_from_vertical_deg": angle_from_vertical_deg,
            "n_candidates": int(len(candidate_idx)),
            "n_inliers": int(len(inlier_orig)),
        }

        return labels_out, side_artifacts

    def get_params(self) -> dict:
        return {
            "ransac_threshold": self._ransac_threshold,
            "max_iterations": self._max_iterations,
            "ransac_n": self._ransac_n,
            "expected_flank_angle_deg": self._expected_flank_angle_deg,
            "normal_cos_threshold": self._normal_cos_threshold,
        }

    def get_artifacts(self) -> dict:
        return {
            k: dict(v) if isinstance(v, dict) else v
            for k, v in self._last_artifacts.items()
        }