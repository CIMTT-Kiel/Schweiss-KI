"""
Label-Definitionen für die Segmentierung (AP2.1 Phase 3).

Single Source of Truth – von segmentation/*, data_structures.py
und Validierungs-Notebooks importiert.

Achsen-Konvention (identisch zur Subtraktions-Stage, siehe
subtraction/deviation/gap_profile.py):
    seam_axis     – Naht-Längsrichtung          (Default 0 = X)
    gap_axis      – Spalt-Querrichtung          (Default 1 = Y)
    vertical_axis – Tiefe, Flanken laufen nach unten (Default 2 = Z)

Die Achsen sind in FlankSegmenter und GapClassifier konfigurierbar; die
Defaults entsprechen den erzeugten Datensätzen (scripts/generate_synthetic_dataset.py
trennt die Werkstücke entlang Y, die Naht läuft also entlang X).

Konvention Flank A / B (relativ zur gap_axis, nicht zu X):
    - Flank A = Fase auf der negativen gap_axis-Seite,
                Normale zeigt in +gap_axis-Richtung (n[gap_axis] > 0)
    - Flank B = Fase auf der positiven gap_axis-Seite,
                Normale zeigt in -gap_axis-Richtung (n[gap_axis] < 0)
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
    1: "V-Fase auf der negativen gap_axis-Seite (Normale n[gap_axis] > 0)",
    2: "V-Fase auf der positiven gap_axis-Seite (Normale n[gap_axis] < 0)",
    3: "Freier Raum zwischen den Fasen, innerhalb der Flanken-Tiefen-Bounds",
    4: "Punkte unterhalb der Flanken-Unterkante (Messartefakte, z.B. Durchstich)",
}

NAME_TO_ID: Final[dict[str, int]] = {v: k for k, v in LABELS.items()}

UNLABELED: Final[int] = -1
"""Initialwert für labels-Array.
Pipeline-Ende konvertiert verbleibende -1 zu 0 (background) als Sicherheitsnetz."""