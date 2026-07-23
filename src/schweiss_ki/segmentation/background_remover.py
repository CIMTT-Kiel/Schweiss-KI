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
        split_gap_axis: int | None = 1,
        split_value: float = 0.0,
        vertical_axis: int = 2,
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
            split_gap_axis:       Achse, an deren Vorzeichen die beiden
                                  Werkstücke getrennt werden (Default 1 = Y).
                                  None = ein gemeinsamer Fit wie bisher.
            split_value:          Trennwert auf split_gap_axis.
            vertical_axis:        Achse für z_center im Report.
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
        self._split_gap_axis = None if split_gap_axis is None else int(split_gap_axis)
        self._split_value = float(split_value)
        self._vertical_axis = int(vertical_axis)
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

        # 2./3. Ebenenfit je Werkstückseite, getrennt am gap_axis-Vorzeichen.
        #
        # Ein gemeinsamer Fit über beide Werkstückoberseiten scheitert, sobald die
        # Werkstücke gegeneinander verkippt sind: RANSAC findet dann die
        # dominante Hälfte und behandelt die andere als Ausreißer. Gemessen an
        # R_Y_+01.000deg passten nur 251.284 von 435.483 Punkten der Werkstückoberseite in
        # die gefundene Ebene – die übrigen 42 % fielen erst am Pipeline-Ende
        # über fill_unlabeled_with_background in Label 0, statt hier sauber
        # klassifiziert zu werden.
        sides = self._split_sides(points, unlabeled_idx, candidate_idx)

        background_idx_all: list[np.ndarray] = []
        side_artifacts: dict = {}
        for side_name, side_cand, side_unlab in sides:
            if len(side_cand) < self._ransac_n:
                logger.debug(
                    f"{self.name} ({side_name}): nur {len(side_cand)} Kandidaten, "
                    f"Seite übersprungen."
                )
                side_artifacts[side_name] = {
                    "status": "insufficient_candidates",
                    "n_candidates": int(len(side_cand)),
                    "n_inliers": 0,
                }
                continue

            normal_unit, d_unit = self._fit_plane(pcd, side_cand)
            distances = np.abs(points[side_unlab] @ normal_unit + d_unit)
            side_bg = side_unlab[distances < self._ransac_threshold]
            background_idx_all.append(side_bg)

            cos_to_expected = abs(normal_unit @ self._expected_normal)
            side_artifacts[side_name] = {
                "status": "ok",
                "plane_model": [float(x) for x in (*normal_unit, d_unit)],
                "plane_normal": normal_unit.tolist(),
                "tilt_angle_deg": float(
                    np.degrees(np.arccos(np.clip(cos_to_expected, -1.0, 1.0)))
                ),
                "z_center": (
                    float(np.mean(points[side_bg, self._vertical_axis]))
                    if len(side_bg) else float("nan")
                ),
                "n_candidates": int(len(side_cand)),
                "n_inliers": int(len(side_bg)),
            }

        background_idx = (
            np.concatenate(background_idx_all) if background_idx_all
            else np.empty(0, dtype=int)
        )
        labels_out[background_idx] = NAME_TO_ID["background"]

        self._last_artifacts = self._build_artifacts(
            side_artifacts, candidate_idx, background_idx
        )
        return labels_out

    # ── Helfer ────────────────────────────────────────────────────────

    def _split_sides(self, points, unlabeled_idx, candidate_idx):
        """Zerlegt Kandidaten und UNLABELED nach gap_axis-Vorzeichen.

        Bei split_gap_axis=None wird nicht getrennt – dann verhält sich der
        Step wie zuvor (ein gemeinsamer Fit).
        """
        if self._split_gap_axis is None:
            return [("combined", candidate_idx, unlabeled_idx)]
        g = self._split_gap_axis
        v = self._split_value
        return [
            ("positive",
             candidate_idx[points[candidate_idx, g] >= v],
             unlabeled_idx[points[unlabeled_idx, g] >= v]),
            ("negative",
             candidate_idx[points[candidate_idx, g] < v],
             unlabeled_idx[points[unlabeled_idx, g] < v]),
        ]

    def _fit_plane(self, pcd, idx):
        """RANSAC-Ebenenfit auf eine Teilmenge, Normale normiert."""
        sub = pcd.select_by_index(idx.tolist())
        model, _ = sub.segment_plane(
            distance_threshold=self._ransac_threshold,
            ransac_n=self._ransac_n,
            num_iterations=self._max_iterations,
        )
        n = np.asarray(model[:3], dtype=float)
        d = float(model[3])
        length = np.linalg.norm(n)
        return n / length, d / length

    def _build_artifacts(self, side_artifacts, candidate_idx, background_idx):
        """Artefakte; die Top-Level-Schlüssel bleiben abwärtskompatibel."""
        ok = [s for s in side_artifacts.values() if s.get("status") == "ok"]
        art: dict = {
            "sides": side_artifacts,
            "n_candidates": int(len(candidate_idx)),
            "n_inliers": int(len(background_idx)),
        }
        if ok:
            # Bisherige Konsumenten (Notebooks, Reports) lesen plane_model,
            # plane_normal, tilt_angle_deg und z_center flach. Belegt wird das
            # mit der Seite, die die meisten Punkte traegt.
            dominant = max(ok, key=lambda s: s["n_inliers"])
            for key in ("plane_model", "plane_normal", "tilt_angle_deg", "z_center"):
                art[key] = dominant[key]
        if len(ok) == 2:
            a, b = (np.asarray(s["plane_normal"]) for s in ok)
            cos = float(np.clip(a @ b, -1.0, 1.0))
            art["relative_tilt_deg"] = float(np.degrees(np.arccos(cos)))
        return art

    def get_params(self) -> dict:
        return {
            "ransac_threshold": self._ransac_threshold,
            "max_iterations": self._max_iterations,
            "ransac_n": self._ransac_n,
            "expected_normal": self._expected_normal.tolist(),
            "normal_cos_threshold": self._normal_cos_threshold,
            "split_gap_axis": self._split_gap_axis,
            "split_value": self._split_value,
            "vertical_axis": self._vertical_axis,
        }

    def get_artifacts(self) -> dict:
        return dict(self._last_artifacts)