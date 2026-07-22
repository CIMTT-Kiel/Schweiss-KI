"""Reproduzierbarkeit: deterministische RANSAC-Ergebnisse über Läufe hinweg.

Hintergrund:
    background_remover und flank_segmenter nutzen Open3Ds segment_plane, also
    RANSAC mit Zufallsstichprobe. Ohne Seed liefert dieselbe Punktwolke bei
    jedem Lauf leicht andere Labels – gemessen bis zu 10.979 von 504.200
    Punkten (2.2 %), was sich als ~0.013 mm Streuung in der Spaltbreite
    niederschlägt. Für die 0.25-mm-Toleranz unkritisch, als Methodik aber
    nicht haltbar: zwei Läufe desselben Bauteils dürfen nicht unterschiedliche
    Messwerte liefern, und ohne feste Werte ist kein Referenzvergleich
    zwischen Läufen definierbar.

Warum pro Modell und nicht einmal global:
    o3d.utility.random.seed() setzt globalen Prozess-Zustand. Einmal beim
    Start gesetzt, hinge der RANSAC des 40. Modells davon ab, wie viele
    Aufrufe vorher passiert sind – ein Batch wäre reproduzierbar, ein einzeln
    zum Debuggen nachgerechnetes Modell ergäbe aber einen anderen Wert als im
    Batch. Deshalb wird vor jedem Modell neu geseedet, mit einem aus
    Basis-Seed und model_id abgeleiteten Wert. Damit ist jedes Modell
    unabhängig von seiner Position im Batch reproduzierbar.

Architektur:
    Wie force_utf8_output() gehört das Seeden an die Einstiegspunkte, NICHT
    in Bibliotheks-Module. Ein Modul, das beim Import globalen RNG-Zustand
    setzt, ist eine böse Überraschung für jeden, der es woanders einbindet.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0
_SEED_MODULUS = 2 ** 31 - 1


def derive_seed(base_seed: int, model_id: str) -> int:
    """Leitet einen stabilen Seed für ein Modell aus Basis-Seed und ID ab.

    Bewusst über einen Hash statt über Addition: base_seed + Laufindex würde
    benachbarte Modelle auf benachbarte Seeds legen und bei zwei Basis-Seeds
    überlappende Sequenzen erzeugen (Basis 0/Modell 5 == Basis 5/Modell 0).
    Für eine spätere Streuungsanalyse über mehrere Basis-Seeds müssen die
    Läufe unabhängig sein.

    blake2b statt der eingebauten hash(): Pythons String-Hash ist über
    PYTHONHASHSEED prozessweise gesalzen und damit zwischen Läufen NICHT
    stabil – genau der Fehler, den diese Funktion verhindern soll.
    """
    payload = f"{int(base_seed)}:{model_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def seed_everything(seed: int) -> int:
    """Setzt alle bekannten globalen RNG-Quellen auf `seed`.

    Open3D ist die eigentlich relevante Quelle (segment_plane/RANSAC). Der
    numpy-Seed ist defensiv: die bekannten Konsumenten (CoarsePCA, ICPFine)
    führen eigene np.random.default_rng(random_seed)-Instanzen und sind davon
    unberührt – falls aber irgendwo der globale numpy-RNG genutzt wird, ist
    er hiermit ebenfalls festgelegt.

    Returns:
        Den gesetzten Seed, zur Protokollierung im Report.
    """
    seed = int(seed) % _SEED_MODULUS
    o3d.utility.random.seed(seed)
    np.random.seed(seed)
    return seed


def seed_for_model(base_seed: int, model_id: str) -> int:
    """Seedet für die Verarbeitung eines bestimmten Modells.

    Returns:
        Den effektiven Seed – gehört in den Report, damit nachvollziehbar
        bleibt, mit welchem Seed ein Ergebnis entstanden ist.
    """
    effective = seed_everything(derive_seed(base_seed, model_id))
    logger.debug(f"  RNG geseedet für '{model_id}': {effective} (Basis {base_seed})")
    return effective
