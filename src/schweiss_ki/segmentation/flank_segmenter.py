"""
Flanken-Segmentierung via zweimal RANSAC (AP2.1 Phase 3).

Segmentiert die beiden schrägen V-Fasen getrennt, mit Normalen-Vorfilter
zur Unterscheidung links (Flank A) vs. rechts (Flank B).

Achsen-Konvention (identisch zur Subtraktions-Stage):
    seam_axis     – Naht-Längsrichtung          (Default 0 = X)
    gap_axis      – Spalt-Querrichtung          (Default 1 = Y)
    vertical_axis – Tiefe                       (Default 2 = Z)

Konvention:
    Flank A = Fase auf der negativen gap_axis-Seite,
              Normale ≈ ( cos(α) entlang +gap_axis, sin(α) entlang +vertical_axis)
    Flank B = Fase auf der positiven gap_axis-Seite,
              Normale ≈ (-cos(α) entlang  gap_axis, sin(α) entlang +vertical_axis)
    α = expected_flank_angle_deg (Winkel der Flanke zur Vertikalen)

    Für die nominelle 90° V-Naht (Öffnungswinkel = 90°, je 45° pro Seite)
    mit Default-Achsen (gap=Y, vertical=Z):
        Flank A expected: (0,  0.707, 0.707)
        Flank B expected: (0, -0.707, 0.707)

Voraussetzung:
    Normalen müssen konsistent "nach außen" orientiert sein (d.h. vom
    Werkstück-Inneren weg). Bei CMM-Scan von oben entspricht das
    n[vertical_axis] > 0 für die Flanken. Liefert das Preprocessing flipped
    Normalen, findet dieser Step 0 Kandidaten → Warnung im Report.

    Ebenso kritisch: passt gap_axis nicht zur tatsächlichen Lage der Naht,
    liegt der maximale cos-Wert der Flanken bei nur cos(2α)-nahen Werten und
    der Step findet 0 Kandidaten, während die Deckfläche fälschlich die
    höchsten cos-Werte liefert. Achsen daher immer gegen die Daten prüfen.
"""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from .base import SegmentationStep, validate_axes
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
            n_candidates, n_inliers, cos_max
    """

    def __init__(
        self,
        ransac_threshold: float = 0.25,
        max_iterations: int = 1000,
        ransac_n: int = 3,
        expected_flank_angle_deg: float = 30.0,
        normal_cos_threshold: float = 0.85,
        seam_axis: int = 0,
        gap_axis: int = 1,
        vertical_axis: int = 2,
        enabled: bool = True,
    ):
        """
        Args:
            ransac_threshold:         Max. Punkt→Ebene-Abstand für Inlier (mm).
                                      Default 0.25mm = Toleranzanforderung AP2.
            max_iterations:           RANSAC-Iterationen.
            ransac_n:                 Min. Punkte pro RANSAC-Hypothese.
            expected_flank_angle_deg: Erwartete Flankenneigung zur Vertikalen
                                      in Grad. 30° = nominelle 60° V-Naht,
                                      45° = 90° V-Naht.
            normal_cos_threshold:     Signed cos-Schwelle für Kandidaten-
                                      Vorfilter. 0.85 ≈ ±32°, 0.9 ≈ ±26°,
                                      0.95 ≈ ±18°.
            seam_axis:                Naht-Längsrichtung (0=X, 1=Y, 2=Z).
            gap_axis:                 Spalt-Querrichtung – die Achse, entlang
                                      derer die Flanken-Normalen auseinander
                                      zeigen.
            vertical_axis:            Tiefenrichtung, Flanken-Normalen haben
                                      hier eine positive Komponente.
            enabled:                  Step aktiv/inaktiv.
        """
        self._ransac_threshold = ransac_threshold
        self._max_iterations = max_iterations
        self._ransac_n = ransac_n
        self._expected_flank_angle_deg = expected_flank_angle_deg
        self._normal_cos_threshold = normal_cos_threshold
        self._seam_axis, self._gap_axis, self._vertical_axis = validate_axes(
            seam_axis, gap_axis, vertical_axis
        )
        self._enabled = enabled

        alpha = np.deg2rad(expected_flank_angle_deg)
        # Flank A: Normale zeigt in +gap_axis, Flank B in -gap_axis;
        # beide mit positiver vertical_axis-Komponente (nach oben/außen).
        self._expected_normal_a = self._build_expected_normal(alpha, +1.0)
        self._expected_normal_b = self._build_expected_normal(alpha, -1.0)

        self._last_artifacts: dict = {}

    def _build_expected_normal(self, alpha: float, gap_sign: float) -> np.ndarray:
        """Erwartete Flanken-Normale im konfigurierten Achsensystem."""
        normal = np.zeros(3)
        normal[self._gap_axis] = gap_sign * np.cos(alpha)
        normal[self._vertical_axis] = np.sin(alpha)
        return normal

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
        cos_max = float(cos_sim.max())

        if len(candidate_idx) < self._ransac_n:
            logger.warning(
                f"{self.name} ({side_name}): nur {len(candidate_idx)} Kandidaten "
                f"nach Normalen-Filter (threshold={self._normal_cos_threshold}, "
                f"cos_max={cos_max:.3f}). Seite übersprungen. "
                f"Prüfen: gap_axis={self._gap_axis}/vertical_axis={self._vertical_axis} "
                f"passend zur Naht-Lage? expected_flank_angle_deg="
                f"{self._expected_flank_angle_deg}° korrekt? Normalen nach außen orientiert?"
            )
            return labels, {
                "status": "insufficient_candidates",
                "n_candidates": int(len(candidate_idx)),
                "n_inliers": 0,
                "cos_max": cos_max,
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

        # Winkel der gefitteten Ebene zur Vertikalen (arcsin(|n[vertical_axis]|))
        angle_from_vertical_deg = float(
            np.degrees(np.arcsin(min(abs(float(normal_unit[self._vertical_axis])), 1.0)))
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
            "cos_max": cos_max,
        }

        return labels_out, side_artifacts

    def get_params(self) -> dict:
        return {
            "ransac_threshold": self._ransac_threshold,
            "max_iterations": self._max_iterations,
            "ransac_n": self._ransac_n,
            "expected_flank_angle_deg": self._expected_flank_angle_deg,
            "normal_cos_threshold": self._normal_cos_threshold,
            "seam_axis": self._seam_axis,
            "gap_axis": self._gap_axis,
            "vertical_axis": self._vertical_axis,
        }

    def get_artifacts(self) -> dict:
        return {
            k: dict(v) if isinstance(v, dict) else v
            for k, v in self._last_artifacts.items()
        }