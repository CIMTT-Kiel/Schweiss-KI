"""
GapProfile – Wurzelspalt-Profil entlang der Naht-Längsrichtung.

Methodik:
    - Für jede der beiden Flanken (Labels A und B) werden die segmentierten
      Punkte entlang der Naht-Längsachse in Bins zerlegt.
    - Pro Bin und Flanke wird eine Geradenanpassung y(z) = m·z + y₀ durchgeführt.
    - Der Y-Schnittpunkt der Geraden mit z = 0 liefert den virtuellen
      Wurzelpunkt der Flanke (die echten Wurzelpunkte fehlen typischerweise,
      weil der CMM-Strahl dort keine Messung mehr aufnimmt).
    - Die Spaltbreite pro Bin ist der Y-Abstand der beiden Wurzelpunkte.

Voraussetzungen:
    - Scan ist bereits ausgerichtet (CoarsePCA + ICPFine vorher in Pipeline).
    - source_labels enthält die Flanken-Labels (typisch 1 und 2 aus AP2.1).
    - Achsen-Konvention: Naht entlang einer konfigurierbaren Achse,
      Spalt entlang einer dazu senkrechten Achse, Vertikale = Z.

Konfigurierbar:
    flank_a_label, flank_b_label     – welche Labels die Flanken bezeichnen
    seam_axis, gap_axis, vertical_axis – Achsen-Konvention (0=X, 1=Y, 2=Z)
    n_bins                             – Anzahl Bins entlang der Naht
    edge_margin                        – mm an beiden Enden ausklammern (Heftpunkte)
    min_points_per_bin                 – Mindest-Punktzahl pro Flanke und Bin
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from ..base import DeviationStep
from ..reports import DeviationData

logger = logging.getLogger(__name__)


class GapProfile(DeviationStep):
    """Wurzelspalt-Profil entlang der Naht-Längsrichtung."""

    def __init__(
        self,
        flank_a_label: int = 1,
        flank_b_label: int = 2,
        seam_axis: int = 0,
        gap_axis: int = 1,
        vertical_axis: int = 2,
        n_bins: int = 20,
        edge_margin: float = 10.0,
        min_points_per_bin: int = 3,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self.flank_a_label = int(flank_a_label)
        self.flank_b_label = int(flank_b_label)
        self.seam_axis = int(seam_axis)
        self.gap_axis = int(gap_axis)
        self.vertical_axis = int(vertical_axis)
        self.n_bins = int(n_bins)
        self.edge_margin = float(edge_margin)
        self.min_points_per_bin = int(min_points_per_bin)

        if len({self.seam_axis, self.gap_axis, self.vertical_axis}) != 3:
            raise ValueError(
                "seam_axis, gap_axis, vertical_axis müssen drei verschiedene "
                f"Achsen sein, sind aber: {self.seam_axis}, {self.gap_axis}, "
                f"{self.vertical_axis}"
            )

    @property
    def name(self) -> str:
        return "gap_profile"

    def get_params(self) -> Dict[str, Any]:
        return {
            "flank_a_label": self.flank_a_label,
            "flank_b_label": self.flank_b_label,
            "seam_axis": self.seam_axis,
            "gap_axis": self.gap_axis,
            "vertical_axis": self.vertical_axis,
            "n_bins": self.n_bins,
            "edge_margin": self.edge_margin,
            "min_points_per_bin": self.min_points_per_bin,
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
        if source_labels is None:
            raise ValueError(
                "GapProfile benötigt source_labels (Flanken-Segmentierung). "
                "AP2.1-Segmentierung muss vorher gelaufen sein."
            )

        pts = np.asarray(source.points)
        flank_a = pts[source_labels == self.flank_a_label]
        flank_b = pts[source_labels == self.flank_b_label]

        if len(flank_a) == 0 or len(flank_b) == 0:
            logger.warning(
                f"  GapProfile: leere Flanken (A={len(flank_a)}, B={len(flank_b)}). "
                f"Step übersprungen."
            )
            return {"n_bins_valid": 0, "skipped": True}

        # Bin-Grenzen entlang der Naht-Achse, Margin gegen Heftpunkte
        seam_min = max(flank_a[:, self.seam_axis].min(),
                       flank_b[:, self.seam_axis].min()) + self.edge_margin
        seam_max = min(flank_a[:, self.seam_axis].max(),
                       flank_b[:, self.seam_axis].max()) - self.edge_margin

        if seam_max <= seam_min:
            logger.warning(
                f"  GapProfile: edge_margin ({self.edge_margin} mm) zu groß für "
                f"verbleibende Naht-Länge. Step übersprungen."
            )
            return {"n_bins_valid": 0, "skipped": True}

        bin_edges = np.linspace(seam_min, seam_max, self.n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        gap_widths = np.full(self.n_bins, np.nan)
        y_a0 = np.full(self.n_bins, np.nan)
        y_b0 = np.full(self.n_bins, np.nan)

        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask_a = (flank_a[:, self.seam_axis] >= lo) & (flank_a[:, self.seam_axis] < hi)
            mask_b = (flank_b[:, self.seam_axis] >= lo) & (flank_b[:, self.seam_axis] < hi)
            sub_a = flank_a[mask_a]
            sub_b = flank_b[mask_b]

            ya = self._extrapolate_to_z0(sub_a)
            yb = self._extrapolate_to_z0(sub_b)

            if not (np.isnan(ya) or np.isnan(yb)):
                gap_widths[i] = abs(ya - yb)
                y_a0[i] = ya
                y_b0[i] = yb

        n_valid = int((~np.isnan(gap_widths)).sum())
        logger.info(
            f"  GapProfile: ausgewertet in {n_valid}/{self.n_bins} Bins, "
            f"Spaltbreite min={np.nanmin(gap_widths):.2f}, "
            f"max={np.nanmax(gap_widths):.2f} mm"
            if n_valid > 0 else f"  GapProfile: keine validen Bins"
        )

        # In DeviationData schreiben
        data.gap_profile = {
            "seam_axis_centers": bin_centers,
            "gap_widths": gap_widths,
            "y_a_root": y_a0,
            "y_b_root": y_b0,
            "seam_axis": self.seam_axis,
            "gap_axis": self.gap_axis,
        }

        artifacts: Dict[str, Any] = {
            "n_bins_total": self.n_bins,
            "n_bins_valid": n_valid,
            "seam_range_used": (float(seam_min), float(seam_max)),
        }
        if n_valid > 0:
            artifacts.update({
                "gap_min_mm": float(np.nanmin(gap_widths)),
                "gap_max_mm": float(np.nanmax(gap_widths)),
                "gap_mean_mm": float(np.nanmean(gap_widths)),
                "gap_std_mm": float(np.nanstd(gap_widths)),
            })
        return artifacts

    # ── Helfer ────────────────────────────────────────────────────────

    def _extrapolate_to_z0(self, pts: np.ndarray) -> float:
        """Lineare Anpassung y(z) = m·z + y0, gibt y0 zurück.

        y = pts[:, gap_axis], z = pts[:, vertical_axis].
        """
        if len(pts) < self.min_points_per_bin:
            return float("nan")
        z = pts[:, self.vertical_axis]
        y = pts[:, self.gap_axis]
        try:
            _, y0 = np.polyfit(z, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            return float("nan")
        return float(y0)