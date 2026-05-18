"""
CoarsePCA – PCA-basierte Grob-Registrierung.

Idee:
    1. Schwerpunkte und Hauptachsen beider Wolken via PCA berechnen.
    2. Source so transformieren, dass seine Hauptachsen mit denen von
       target übereinstimmen (Schwerpunkt → Schwerpunkt, Achsen → Achsen).
    3. PCA hat eine 4-fache Vorzeichen-Ambiguität (jede Achse kann
       umgedreht sein, die dritte muss det = +1 erhalten). Wir probieren
       alle 4 gültigen Kombinationen und nehmen die mit dem niedrigsten
       mittleren NN-Abstand.

Was bleibt für ICP:
    - Feine Rotation (PCA ist nur grob, ~1-3° Restfehler typisch)
    - Feine Translation auf der Naht-Ebene
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from ..base import RegistrationStep

logger = logging.getLogger(__name__)


class CoarsePCA(RegistrationStep):
    """PCA-Achsen-Ausrichtung als Grob-Registrierung.

    Args:
        anchor_labels:    Wenn gegeben, wird PCA nur auf Punkten mit diesen
                          Labels berechnet (typisch: {0, 1, 2} – Anker-Regionen,
                          Spalt ausgeschlossen). Default: None = alle Punkte.
        evaluation_samples: Anzahl Source-Punkte für das Kandidaten-Ranking
                          (mittlerer NN-Abstand). Default 5000, mehr = genauer
                          aber langsamer.
        enabled:          Step-Aktivierung.
        random_seed:      Seed für deterministisches Subsampling im Ranking.
    """

    def __init__(
        self,
        anchor_labels: Optional[Sequence[int]] = None,
        evaluation_samples: int = 5_000,
        enabled: bool = True,
        random_seed: int = 0,
    ):
        self._enabled = enabled
        self.anchor_labels = (
            None if anchor_labels is None else tuple(anchor_labels)
        )
        self.evaluation_samples = int(evaluation_samples)
        self.random_seed = int(random_seed)

    @property
    def name(self) -> str:
        return "coarse_pca"

    def get_params(self) -> Dict[str, Any]:
        return {
            "anchor_labels": self.anchor_labels,
            "evaluation_samples": self.evaluation_samples,
            "random_seed": self.random_seed,
        }

    # ── Hauptlogik ────────────────────────────────────────────────────

    def _apply(
        self,
        source_aligned: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        src_pts = self._anchor_points(source_aligned, source_labels)
        tgt_pts = self._anchor_points(target, target_labels)

        if len(src_pts) < 3 or len(tgt_pts) < 3:
            raise ValueError(
                f"Zu wenig Anker-Punkte (source={len(src_pts)}, "
                f"target={len(tgt_pts)}). Mindestens 3 nötig."
            )

        src_center, src_axes, src_eigvals = self._pca(src_pts)
        tgt_center, tgt_axes, tgt_eigvals = self._pca(tgt_pts)

        # 4 gültige Vorzeichen-Kombinationen (alle mit Produkt +1)
        flip_options = [
            (+1, +1, +1),
            (-1, -1, +1),
            (-1, +1, -1),
            (+1, -1, -1),
        ]

        # Source-Subsample für die Kandidaten-Bewertung
        all_src = np.asarray(source_aligned.points)
        n_eval = min(self.evaluation_samples, len(all_src))
        rng = np.random.default_rng(self.random_seed)
        eval_idx = rng.choice(len(all_src), n_eval, replace=False)
        src_sample = all_src[eval_idx]

        # KDTree auf target einmal bauen
        kdtree = o3d.geometry.KDTreeFlann(target)

        candidates = []
        for flip in flip_options:
            src_axes_flipped = src_axes * np.array(flip)  # spaltenweise flip
            R = tgt_axes @ src_axes_flipped.T
            t = tgt_center - R @ src_center

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            cost = self._mean_nn_to_target(src_sample, T, kdtree)
            candidates.append((cost, T, flip))

        candidates.sort(key=lambda x: x[0])
        best_cost, best_T, best_flip = candidates[0]

        logger.debug(
            f"  CoarsePCA Kandidaten (mittlerer NN nach Transform):\n"
            + "\n".join(
                f"    flip={c[2]}: {c[0]:.3f} mm" for c in candidates
            )
        )

        artifacts = {
            "anchor_count_source": len(src_pts),
            "anchor_count_target": len(tgt_pts),
            "source_centroid": src_center.tolist(),
            "target_centroid": tgt_center.tolist(),
            "source_eigvals": src_eigvals.tolist(),
            "target_eigvals": tgt_eigvals.tolist(),
            "flip_chosen": best_flip,
            "candidate_costs": {str(c[2]): float(c[0]) for c in candidates},
            "fitness": float(1.0 / (1.0 + best_cost)),  # informativ, 1 = perfekt
        }

        return best_T, artifacts

    def compute_residual(
        self,
        source_after: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Mittlerer NN-Abstand source → target nach Anwendung der Transform."""
        all_src = np.asarray(source_after.points)
        n_eval = min(self.evaluation_samples, len(all_src))
        rng = np.random.default_rng(self.random_seed)
        idx = rng.choice(len(all_src), n_eval, replace=False)
        kdtree = o3d.geometry.KDTreeFlann(target)
        return self._mean_nn_to_target(
            all_src[idx], np.eye(4), kdtree
        )

    # ── Helfer ────────────────────────────────────────────────────────

    def _anchor_points(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: Optional[np.ndarray],
    ) -> np.ndarray:
        pts = np.asarray(pcd.points)
        if self.anchor_labels is None or labels is None:
            return pts
        mask = np.isin(labels, self.anchor_labels)
        return pts[mask]

    @staticmethod
    def _pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Schwerpunkt + Hauptachsen (3×3, Spalten = Achsen, absteigend) + Eigenwerte."""
        center = points.mean(axis=0)
        centered = points - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        axes = eigvecs[:, order]
        # Rechtshändig erzwingen (det = +1)
        if np.linalg.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]
        return center, axes, eigvals

    @staticmethod
    def _mean_nn_to_target(
        src_pts: np.ndarray,
        T: np.ndarray,
        kdtree: o3d.geometry.KDTreeFlann,
    ) -> float:
        """Mittlerer NN-Abstand von T @ src_pts zum target (kdtree)."""
        R, t = T[:3, :3], T[:3, 3]
        moved = src_pts @ R.T + t
        sq_dists = np.empty(len(moved))
        for i, p in enumerate(moved):
            _, _, d2 = kdtree.search_knn_vector_3d(p, 1)
            sq_dists[i] = d2[0]
        return float(np.sqrt(sq_dists).mean())