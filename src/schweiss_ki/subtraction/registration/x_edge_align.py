"""
XEdgeAlign – X-Vorregistrierung über Bauteil-Kantenpositionen.

Motivation:
    ICP versagt bei Verschiebungen entlang der Naht-Längsachse (X),
    weil die Geometrie entlang dieser Achse weitgehend uniform ist. Die
    einzige X-Information sitzt an den Bauteil-Enden. Bei
    Verschiebungen > max_correspondence_distance hat kein Randpunkt eine
    Korrespondenz innerhalb des ICP-Suchradius, sodass die Registrierung
    nur so weit zieht wie der Radius erlaubt.

    Dieser Step schätzt die X-Verschiebung direkt aus den Kantenpositionen
    und wendet sie als reine Δx-Translation an. Danach greift der enge
    ICP-Radius wieder.

Verfahren:
    1. Punkte auf Anker-Labels beschränken (Default: [0, 1, 2]).
    2. In X-Bins fester Breite rastern.
    3. Von außen nach innen den ersten Bin mit ausreichender Belegung
       suchen. Kantenposition = Median der X-Werte dieses Bins.
    4. Δx pro Seite = Target-Kante − Source-Kante.
    5. Aggregation:
       - Beide Seiten valide  → Mittelwert der beiden Δx
       - Nur eine Seite       → deren Δx allein (partial observability)
       - Keine Seite          → identity (nicht beobachtbar)

Diagnostik-Ausgaben im Artefakt:
    - observability:    "full" | "partial_left" | "partial_right" | "none"
    - consistency_mm:   Δx_links − Δx_rechts (nur bei "full")
    - length_source:    Bauteillänge in X aus dem Source
    - length_target:    Bauteillänge in X aus dem Target
    - length_diff:      Differenz der Längen als Diagnostik-Signal
    - edge positions:   die vier gefundenen Kantenpositionen einzeln
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from ..base import RegistrationStep
from ..reports import RegistrationStepReport

logger = logging.getLogger(__name__)


class XEdgeAlign(RegistrationStep):
    """X-Vorregistrierung über Bauteil-Kantenpositionen."""

    def __init__(
        self,
        bin_width_mm: float = 0.5,
        min_points_per_bin: int = 20,
        anchor_labels: Optional[List[int]] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self.bin_width_mm = float(bin_width_mm)
        self.min_points_per_bin = int(min_points_per_bin)
        self.anchor_labels = tuple(anchor_labels) if anchor_labels is not None else (0, 1, 2)

    @property
    def name(self) -> str:
        return "x_edge_align"

    def get_params(self) -> Dict[str, Any]:
        return {
            "bin_width_mm": self.bin_width_mm,
            "min_points_per_bin": self.min_points_per_bin,
            "anchor_labels": list(self.anchor_labels),
        }

    # ── Hauptlogik ────────────────────────────────────────────────────

    def _apply(
        self,
        source_aligned: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        source_pts = np.asarray(source_aligned.points)
        target_pts = np.asarray(target.points)

        # Source auf Anker-Labels beschränken (falls Labels vorhanden)
        if source_labels is not None:
            anchor_mask = np.isin(source_labels, self.anchor_labels)
            source_x = source_pts[anchor_mask, 0]
            n_source_used = int(anchor_mask.sum())
        else:
            source_x = source_pts[:, 0]
            n_source_used = len(source_pts)

        target_x = target_pts[:, 0]

        # Kantenposition pro Seite finden
        source_left, source_left_valid = self._find_edge(source_x, from_left=True)
        source_right, source_right_valid = self._find_edge(source_x, from_left=False)
        target_left, target_left_valid = self._find_edge(target_x, from_left=True)
        target_right, target_right_valid = self._find_edge(target_x, from_left=False)

        left_valid = source_left_valid and target_left_valid
        right_valid = source_right_valid and target_right_valid

        delta_left = target_left - source_left if left_valid else None
        delta_right = target_right - source_right if right_valid else None

        # Aggregation + Observability-Flag
        if left_valid and right_valid:
            delta_x = (delta_left + delta_right) / 2.0
            observability = "full"
            consistency_mm = float(delta_left - delta_right)
        elif left_valid:
            delta_x = delta_left
            observability = "partial_left"
            consistency_mm = None
        elif right_valid:
            delta_x = delta_right
            observability = "partial_right"
            consistency_mm = None
        else:
            delta_x = 0.0
            observability = "none"
            consistency_mm = None

        # Diagnostik: Bauteillänge
        length_source = float(source_x.max() - source_x.min()) if len(source_x) > 0 else 0.0
        length_target = float(target_x.max() - target_x.min()) if len(target_x) > 0 else 0.0
        length_diff = length_source - length_target

        # Delta-Transformation als 4×4 mit Δx im Translationsanteil
        delta_transform = np.eye(4)
        delta_transform[0, 3] = delta_x

        artifacts = {
            "delta_x_mm": float(delta_x),
            "observability": observability,
            "consistency_mm": consistency_mm,
            "delta_left_mm": float(delta_left) if delta_left is not None else None,
            "delta_right_mm": float(delta_right) if delta_right is not None else None,
            "edges": {
                "source_left_mm": float(source_left) if source_left_valid else None,
                "source_right_mm": float(source_right) if source_right_valid else None,
                "target_left_mm": float(target_left) if target_left_valid else None,
                "target_right_mm": float(target_right) if target_right_valid else None,
            },
            "length_source_mm": length_source,
            "length_target_mm": length_target,
            "length_diff_mm": float(length_diff),
            "n_source_used": n_source_used,
        }

        logger.info(
            f"  XEdgeAlign: Δx={delta_x:+.4f} mm ({observability})"
            + (f", consistency={consistency_mm:+.4f} mm" if consistency_mm is not None else "")
            + f", length_diff={length_diff:+.4f} mm"
        )

        return delta_transform, artifacts

    # ── Helpers ───────────────────────────────────────────────────────

    def _find_edge(
        self, x_values: np.ndarray, from_left: bool = True
    ) -> Tuple[float, bool]:
        """Sucht die Kantenposition per Bin-Scan.

        Rastert x_values in Bins der Breite bin_width_mm und läuft von
        außen nach innen. Der erste Bin mit mindestens
        min_points_per_bin Belegung liefert die Kante (Median seiner
        X-Werte).

        Returns:
            (edge_position, valid): Wenn kein Bin die Schwelle erreicht,
            wird (nan, False) zurückgegeben.
        """
        if len(x_values) == 0:
            return float("nan"), False

        x_min, x_max = x_values.min(), x_values.max()
        n_bins = max(1, int(np.ceil((x_max - x_min) / self.bin_width_mm)))

        # Bin-Grenzen: [x_min, x_min + bin_width, ..., x_min + n_bins*bin_width]
        edges = x_min + np.arange(n_bins + 1) * self.bin_width_mm

        bin_range = range(n_bins) if from_left else range(n_bins - 1, -1, -1)

        for i in bin_range:
            lo, hi = edges[i], edges[i + 1]
            mask = (x_values >= lo) & (x_values < hi)
            n = int(mask.sum())
            if n >= self.min_points_per_bin:
                return float(np.median(x_values[mask])), True

        return float("nan"), False