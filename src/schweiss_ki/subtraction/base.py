"""
Abstrakte Basisklassen für das Subtraction-Modul (AP2.2).

Analog zu schweiss_ki.preprocessing.base.PreprocessingStep und
schweiss_ki.segmentation.base.SegmentationStep:

- RegistrationStep: produziert eine Delta-Transformation
- DeviationStep:    trägt zur DeviationData bei

Beide Step-Typen:
  - Müssen `name`, `enabled` und `_apply()` implementieren
  - Werden über `__call__()` aufgerufen, das Timing + Report-Erzeugung kapselt
  - Liefern `get_params()` für die Report-Serialisierung
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d

from .reports import DeviationData, DeviationStepReport, RegistrationStepReport

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# RegistrationStep
# ─────────────────────────────────────────────────────────────────────────

class RegistrationStep(ABC):
    """Abstrakte Basisklasse für Registrierungs-Schritte.

    Ein Step bekommt source (= bereits durch vorherige Steps transformierter
    Scan) und target (= CAD-Referenz) und gibt die DELTA-Transformation
    zurück, die dieser Step beiträgt. Die Pipeline akkumuliert die Deltas.

    Konvention:
        - source (Scan) wird ausgerichtet; target (CAD) bleibt fest.
        - source_aligned beim Step-Aufruf bereits durch vorherige Transforms bewegt.
        - Rückgabe: delta_transform, sodass nach diesem Step gilt
          source_neu = delta_transform @ source_aligned
    """

    # ── Pflicht-Interface ─────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Step-Name (snake_case)."""

    @property
    def enabled(self) -> bool:
        """Ob der Step ausgeführt werden soll. Default: True."""
        return getattr(self, "_enabled", True)

    @abstractmethod
    def _apply(
        self,
        source_aligned: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Implementiert die Step-Logik.

        Args:
            source_aligned:  Bereits durch vorherige Steps transformierter Scan.
            target:          CAD-Referenz (fest).
            source_labels:   Optionale Segmentierungs-Labels für source.
            target_labels:   Optionale Segmentierungs-Labels für target.

        Returns:
            (delta_transform, artifacts): 4×4 NumPy-Matrix + Step-spezifische Daten
            (z.B. Inlier-Indizes, gefittete Ebenen, Korrespondenzen).
        """

    def get_params(self) -> Dict[str, Any]:
        """Step-Parameter für den Report. Override empfohlen."""
        return {}

    # ── Optional: Residual/Fitness ────────────────────────────────────

    def compute_residual(
        self,
        source_after: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Berechnet ein Residuum-Maß (z.B. RMSE) nach Anwendung des Steps.

        Default: None. Steps können das überschreiben, wenn sie ein
        sinnvolles Maß berechnen können.
        """
        return None

    # ── Aufruf-Wrapper ────────────────────────────────────────────────

    def __call__(
        self,
        source_aligned: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> RegistrationStepReport:
        """Führt den Step aus und liefert den Report zurück.

        Note:
            Der Step modifiziert source_aligned NICHT in-place. Die Pipeline
            ist dafür verantwortlich, die Delta-Transform anschließend
            anzuwenden.
        """
        if not self.enabled:
            return RegistrationStepReport(
                step_name=self.name,
                enabled=False,
                duration_ms=0.0,
                params=self.get_params(),
            )

        t0 = time.perf_counter()
        delta, artifacts = self._apply(
            source_aligned, target, source_labels, target_labels
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(delta, np.ndarray) or delta.shape != (4, 4):
            raise ValueError(
                f"{self.name}._apply() muss eine 4x4 numpy-Matrix als "
                f"erstes Element des Tuples zurückgeben, bekam {type(delta)} "
                f"mit shape {getattr(delta, 'shape', '?')}"
            )

        # Residuum nach Anwendung des Steps (optional)
        source_after = o3d.geometry.PointCloud(source_aligned)
        source_after.transform(delta)
        residual = self.compute_residual(
            source_after, target, source_labels, target_labels
        )
        fitness = artifacts.pop("fitness", None) if isinstance(artifacts, dict) else None

        return RegistrationStepReport(
            step_name=self.name,
            enabled=True,
            duration_ms=duration_ms,
            delta_transform=delta,
            residual=residual,
            fitness=fitness,
            params=self.get_params(),
            artifacts=artifacts if isinstance(artifacts, dict) else {},
        )


# ─────────────────────────────────────────────────────────────────────────
# DeviationStep
# ─────────────────────────────────────────────────────────────────────────

class DeviationStep(ABC):
    """Abstrakte Basisklasse für Differenzanalyse-Schritte.

    Ein Step bekommt die bereits ausgerichtete Wolke und schreibt
    Ergebnisse in das gemeinsame DeviationData-Objekt
    (signed distances, per-region metrics, gap profile, ...).

    Konvention:
        - source (Scan) ist bereits final ausgerichtet.
        - target (CAD) ist fest.
        - Steps mutieren das übergebene DeviationData direkt.
        - Welche Felder ein Step schreibt, dokumentiert er in seinem Docstring.
    """

    # ── Pflicht-Interface ─────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Step-Name (snake_case)."""

    @property
    def enabled(self) -> bool:
        return getattr(self, "_enabled", True)

    @abstractmethod
    def _apply(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        data: DeviationData,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Implementiert die Step-Logik.

        Args:
            source:         Bereits ausgerichteter Scan.
            target:         CAD-Referenz.
            data:           Gemeinsames DeviationData-Objekt – wird vom Step
                            befüllt/aktualisiert.
            source_labels:  Optionale Source-Labels.
            target_labels:  Optionale Target-Labels.

        Returns:
            Artifacts-Dict (Step-spezifische Zwischenergebnisse).
        """

    def get_params(self) -> Dict[str, Any]:
        return {}

    # ── Aufruf-Wrapper ────────────────────────────────────────────────

    def __call__(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        data: DeviationData,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> DeviationStepReport:
        if not self.enabled:
            return DeviationStepReport(
                step_name=self.name,
                enabled=False,
                duration_ms=0.0,
                params=self.get_params(),
            )

        t0 = time.perf_counter()
        artifacts = self._apply(source, target, data, source_labels, target_labels)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        return DeviationStepReport(
            step_name=self.name,
            enabled=True,
            duration_ms=duration_ms,
            params=self.get_params(),
            artifacts=artifacts if isinstance(artifacts, dict) else {},
        )