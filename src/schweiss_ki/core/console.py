"""Konsolen-Ausgabe plattformunabhängig auf UTF-8 stellen.

Hintergrund – bekannte Fehlerklasse dieser Codebasis:
    Der Code nimmt an mehreren Stellen UTF-8 an. Auf dem Linux-Rechner, auf
    dem das Projekt entstanden ist, stimmt das. Unter Windows ist die
    Locale-Encoding cp1252, und jede Ausgabe mit Nicht-ASCII bricht ab:

        UnicodeEncodeError: 'charmap' codec can't encode character '\\u2713'

    Bisher in dieser Klasse aufgetreten:
      1. YAML-Config-Loader (open ohne encoding)      -> Datei-Lesen
      2. JSON-Report-Loader/Writer (dito)             -> Datei-Schreiben
      3. print/logging mit Umlauten und Symbolen      -> stdout/stderr

    Punkt 3 fällt nur auf, wenn stdout UMGELEITET wird: an einer UTF-8-fähigen
    Konsole läuft es durch, in `> log.txt` nicht. Ein Scan über den Code
    findet 112 Ausgabe-Statements mit Nicht-ASCII in 17 Dateien – die Zeichen
    zu entfernen wäre der falsche Fix (es ist deutscher Fachtext), die Streams
    umzustellen der richtige.

Verwendung: als erste Zeile in jedem Einstiegspunkt unter scripts/.
Bibliotheks-Module brauchen es nicht: deren logging-Ausgabe fließt über die
Handler des Einstiegspunkts und ist damit mit abgedeckt.
"""
from __future__ import annotations

import sys


def force_utf8_output() -> None:
    """Stellt stdout und stderr auf UTF-8 um, auch bei Umleitung.

    Idempotent und ohne Effekt, wenn der Stream bereits UTF-8 spricht oder
    kein reconfigure() anbietet (z.B. bei umgeleiteten Test-Captures).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream nicht umkonfigurierbar (bereits detached o.ä.) – die
            # Ausgabe ist dann bestenfalls unvollständig, aber nichts bricht.
            pass
