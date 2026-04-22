"""
Label-Definitionen für die Segmentierung (AP2.1 Phase 3).

Single Source of Truth – von segmentation/*, data_structures.py
und Validierungs-Notebooks importiert.

Konvention Flank A / B:
    - Flank A = linke V-Fase  (Normale zeigt nach rechts-oben, n_x > 0)
    - Flank B = rechte V-Fase (Normale zeigt nach links-oben,  n_x < 0)

Koordinatensystem: X = Stirnseite (quer zur Naht), Y = Nahtrichtung, Z = Tiefe
"""
from __future__ import annotations

from typing import Final

LABELS: Final[dict[int, str]] = {
    0: "background",
    1: "flank_a",
    2: "flank_b",
    3: "gap_region",
    4: "sub_gap_artifacts",
}

LABEL_DESCRIPTIONS: Final[dict[int, str]] = {
    0: "Werkstück-Oberseite, Umgebung, Messartefakte außerhalb der Naht",
    1: "Linke V-Fase (Normale n_x > 0)",
    2: "Rechte V-Fase (Normale n_x < 0)",
    3: "Freier Raum zwischen den Fasen, innerhalb der Flanken-Z-Bounds",
    4: "Punkte unterhalb der Flanken-Unterkante (Messartefakte, z.B. Durchstich)",
}

NAME_TO_ID: Final[dict[str, int]] = {v: k for k, v in LABELS.items()}

UNLABELED: Final[int] = -1
"""Initialwert für labels-Array.
Pipeline-Ende konvertiert verbleibende -1 zu 0 (background) als Sicherheitsnetz."""