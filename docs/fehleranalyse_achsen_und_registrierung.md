# Fehleranalyse: Segmentierung fand 0 Flanken-Kandidaten

**Stand:** 2026-07-22 · **Branch:** `fix/achsen-konvention-und-registrierung`
**Datensatz:** 61 synthetische Fälle, Bauteil `Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt`
(199 × 98 × 5 mm, 90°-V-Naht, 1.5 mm Wurzelspalt)

---

## 1. Ausgangssymptom

Der `FlankSegmenter` fand in **allen 61 synthetischen Fällen 0 Kandidaten**. Die
resultierende Punktwolke trug durchgehend Label 0 (`background`).

Die naheliegende Vermutung — der Segmenter laufe nur auf `labels == UNLABELED`,
die Wolke sei aber schon vollständig gelabelt — war **falsch**. `UNLABELED` ist
`-1`, nicht `0`. Die Pipeline startet mit `np.full(n, -1)` und konvertiert
verbliebene `-1` erst **am Ende** zu `0`:

```python
# segmentation/base.py
if self.fill_unlabeled_with_background:
    labels[labels == UNLABELED] = 0
```

Das durchgehende Label 0 war also die **Folge** des Fehlers, nicht die Ursache.
Weil beide Steps nach dem `background_remover` nichts fanden, blieb alles
`UNLABELED` und wurde pauschal zu Background.

---

## 2. Ursache: Achsen-Konvention

`FlankSegmenter` erwartete Flanken-Normalen mit horizontaler Komponente entlang
**X**, die Datensätze trennen die Werkstücke aber entlang **Y** (Naht läuft
entlang X). Gemessen am echten CAD:

| Fläche | Normale | cos mit erwarteter Normale |
|---|---|---|
| echte Flanke | `[0, ±0.71, 0.71]` | **0.500** |
| Deckfläche | `[0, 0, 1]` | **0.707** |

Bei `normal_cos_threshold = 0.85` passiert keine der beiden den Vorfilter.

**Die Falle:** Der beobachtete `cos_max` von ~0.77 stammte von der verrauschten
**Deckfläche**, nicht von den Flanken (die kommen nie über 0.50). Ein Senken des
Schwellwerts auf ~0.7 hätte deshalb die Werkstück-Oberseite als Flanke gelabelt
und die echten Flanken weiterhin verfehlt — belegt durch
`test_lowering_threshold_would_mislabel_top_surface`: >90 % der so gefundenen
Punkte sind Background.

Die Subtraktions-Stage benutzte bereits die richtige Konvention
(`seam_axis: 0`, `gap_axis: 1`). Nur `segmentation/` stand quer dazu — inklusive
der Docstrings in `labels.py`, die die falsche Konvention als verbindlich
beschrieben.

**Behebung** (`22858bc`): `seam_axis` / `gap_axis` / `vertical_axis` in beiden
Steps konfigurierbar, Semantik und Defaults identisch zur Subtraktions-Stage.
Gemeinsame Prüfung über `validate_axes()`. Achsneutrale Namen
(`x_margin` → `gap_margin`, `gap_width_by_y` → `gap_width_by_seam`, …).

**Ergebnis:** 61/61 Fälle mit beiden Flanken, `cos_max = 1.000`.

---

## 3. Was die Analyse zusätzlich freigelegt hat

Die Kette zerfiel in mehrere unabhängige Probleme. Reihenfolge = Reihenfolge der
Entdeckung.

### 3.1 Plattform-Migration Linux → Windows (`ec997cf`, `f72951e`)

Zwei Bugklassen, beide auf Linux unsichtbar:

**Text-Encoding.** `open()` und `Path.read_text()` ohne `encoding=` nutzen die
Locale-Encoding. Unter Windows (cp1252) brach jeder Pfad ab, der
`configs/pipeline.yaml` liest — die Datei enthält Umlaute in Kommentaren. Elf
Stellen betroffen, über sechs Aufrufformen verteilt (`open`, `Path.open`,
`read_text`, `write_text`, `csv`, `json`). Die Tests liefen durch, weil sie
ASCII-Configs nach `tmp_path` schreiben.

**Case-insensitives Glob.** `process_directory` globbte `*.step` **und**
`*.STEP` und hängte die Ergebnisse aneinander. Auf Windows/macOS matchen beide
dieselbe Datei — jede STEP wurde doppelt konvertiert und verarbeitet.

### 3.2 `coarse_pca` zerstörte die Ausrichtung (`a99a8bf`)

Die Grob-Registrierung verschob systematisch um ~2.5 mm in Z. Gemessen an
`T_X_+00.100mm` (Ground Truth: nur `tx=+0.1mm`, Registrierung müsste die
Identität finden):

| | mittlere Distanz Scan → CAD |
|---|---|
| ohne Registrierung | **0.049 mm** |
| mit `final_transform` | **1.721 mm** |

Ursache: Bei einer flachen Platte (199 × 98 × **5** mm) ist die dritte
Hauptkomponente numerisch schlecht konditioniert; die Kandidaten-Auswahl greift
daneben. ICP startet aus falscher Lage und findet nur ein lokales Minimum
(`fitness = 0.30`).

Folge für die Spaltmessung: Die um 2.5 mm angehobene Wolke legte die
Extrapolation auf z=0 **unter den V-Scheitel**, wo sich die Flanken bereits
gekreuzt haben. Die gemessene Breite war dadurch **invertiert** — Steigung −1.04
statt +1.0.

Deaktiviert, nicht entfernt. Begründung steht am Step in der Config.

### 3.3 Synthetische Scans enthielten die Blech-Unterseite (`47f7c46`)

Es gibt zwei Ablageorte für konvertiertes CAD:

| Pfad | Inhalt |
|---|---|
| `data/outputs/cad/<stem>/` | rohe API-Ausgabe, echte CAD-Normalen |
| `data/outputs/<stem>/` | vorverarbeitetes Modell nach `model.save()` |

Der Generator braucht die **rohe**. Das Preprocessing dreht die Normalen per
`orient_mode: camera` sämtlich nach oben, die Unterseite bekommt `n_z = +1`
statt `−1`:

| | Punkte mit `n_z < −0.5` |
|---|---|
| rohe Wolke | 483.494 |
| nach Preprocessing | 563 |

`filter_top_surface(n_z > 0.5)` kann die Unterseite dann nicht mehr
aussortieren. Bei falschem Pfad bestehen **49.6 %** der Scan-Punkte aus
Unterseite, die ein CMM-Scan von oben nie sieht — ohne Korrespondenz im
CAD-Top-Target.

Der Default zeigte auf den richtigen Pfad; die Verwechslung passiert über
`--cad-cache-dir`. Deshalb kein Default-Wechsel, sondern ein **aktiver Schutz**
in `load_cad()`: Bricht ab, wenn weniger als 5 % der Normalen nach unten zeigen.

### 3.4 Voxel-Downsampling verzerrt die Breitenmessung (`e680744`)

`GapClassifier._compute_gap_width_by_seam` misst über `.max()` / `.min()` — reine
Randstatistiken. Voxel-Downsampling ersetzt Punkte durch Voxel-Schwerpunkte und
zieht den äußersten Flankenrand nach innen:

| | Steigung Spaltbreite über `ty` (Soll 1.0) |
|---|---|
| rohe Scans | **1.0000** |
| mit `voxel_grid_downsampler 0.5` | **0.9757** |

Für synthetische Daten ist das Preprocessing jetzt komplett abgeschaltet
(`source_type: synthetic`). **Bei realen Daten kommt der Bias zurück**, sobald
aus Performance- oder Dichtegründen downgesampelt wird — dort aber vom
Messrauschen maskiert. Siehe Abschnitt 5.

---

## 4. Der Verstärkungsmechanismus der Spaltmessung

Nach allen obigen Fixes blieb eine Untererfassung von ~3.6 %. Zwei
Fehlzuordnungen meinerseits, beide später korrigiert:

- **nicht** die Extrapolationsmethode — `_extrapolate_to_z0` ist ein
  Least-Squares-Fit über alle Flankenpunkte und liefert auf unregistrierten
  Rohdaten **exakt** Steigung 1.0000 bei Fehlern < 0.001 mm
- **nicht** die Flanken-Paarung oder die Segmentierung — beides gemessen
  ausgeschlossen (siehe unten)

Die tatsächliche Ursache ist die **Auswertungshöhe**:

> Die Extrapolation zielt auf `z = 0` im Koordinatensystem der *übergebenen*
> Wolke. Ist sie registriert, verschiebt ein Registrierungs-Versatz `dz` die
> Auswertungshöhe. Eine V-Naht öffnet sich mit `dw/dz = 2·tan(α)`.

Für die 90°-Naht (α = 45°) also:

```
Fehler_Spaltbreite = −2 · dz_Registrierung
```

Verifiziert gegen dieselbe Methode auf unregistrierten Rohdaten:

| Kategorie | n | Korrelation | Rest-RMS |
|---|---|---|---|
| `translation_y` | 10 | 0.99998 | 0.0004 mm |
| `translation_z` | 4 | 0.99999 | 0.0077 mm |
| `rotation_x` | 10 | 0.99970 | 0.0141 mm |
| `translation_combo` | 5 | 0.99978 | 0.0068 mm |

Bei `rotation_x` erzeugt das Fehler bis **0.94 mm**, obwohl der *wahre* Spalt
sich kaum ändert (1.475 … 1.528) — die Registrierung allein produziert die
gesamte scheinbare Abweichung.

`rotation_y` folgt demselben Gesetz mit **positionsabhängigem** `dz`: eine
Restrotation kippt die Auswertungsebene, der Höhenfehler wächst linear entlang
der Naht (`dz_eff = dz − x̄·sin(ry_reg)`):

| ry | Vorhersage | tatsächlich |
|---|---|---|
| 0.10° | +0.174 | +0.174 |
| 0.25° | +0.435 | +0.432 |
| 0.50° | +0.876 | +0.849 |

**Der Faktor ist nahtspezifisch:**

| Öffnung | Flanke α | Faktor 2·tan(α) |
|---|---|---|
| 60° | 30° | 1.15 |
| 75° | 37.5° | 1.53 |
| **90°** | **45°** | **2.00** |
| 100° | 50° | 2.38 |

Je spitzer die Naht, desto stärker verstärkt sie Registrierungsfehler.

---

## 5. Verbleibende Einschränkungen

### 5.1 `ry ≥ 1.0°`: Registrierung bricht qualitativ ein

Ab 1.0° Rotation um die Spalt-Querachse findet die Registrierung ein anderes
lokales Minimum (`dz` springt von ~0.000 auf −0.244, ICP-Residuum 0.482). Das
`−2·dz`-Modell trifft dort nicht mehr (+1.425 vorhergesagt vs. +0.355
tatsächlich). Betrifft auch alle Kombinationsfälle mit `ry`-Anteil.

**Ausgeschlossen als Ursache — gemessen, nicht vermutet:** weder Segmentierung
noch Flanken-Paarung. Über `ry = 0 … 1.0°` bleiben die
`FlankSegmenter`-Kandidatenzahlen stabil (33.545 → 31.938, −5 %) und alle 20
Naht-Bins sehen durchgehend **beide** Flanken — kein einziger einseitiger Slice.

Solange `ry` in der Praxis klein bleibt, ist das eine dokumentierte
Verfahrensgrenze, kein Blocker.

### 5.2 `coarse_pca` fehlt für Scans in beliebiger Lage

Aktuell deaktiviert. Synthetische Fälle und eingemessene CMM-Scans liegen
bereits im CAD-Frame. Kommt später ein Scan in freier Lage, muss `coarse_pca`
**repariert** werden — die schlecht konditionierte dritte Hauptachse gesondert
behandeln —, nicht nur zugeschaltet.

Ansatz für später: Z-Lage über die Deckflächen-Ebene fixieren, die der
`background_remover` per RANSAC ohnehin fittet (`plane_model`, `z_center` liegen
im Report). PCA übernähme dann nur die beiden gut konditionierten Achsen in der
Blechebene. Ungetestet.

### 5.3 Downsampling-Bias bei realen Daten

Siehe 3.4. Der Bias verschwindet nicht, er wird nur unsichtbar. Robuster Ersatz
für `.max()` / `.min()`: Perzentile (99./1.) oder ein Flankenebenen-Fit je
Slice, der nicht an einzelnen Randpunkten hängt.

### 5.4 Auswertungshöhe verankern — verlagert die Abhängigkeit

Der naheliegende Fix für Abschnitt 4 ist, die Auswertungshöhe an der
RANSAC-Deckflächenebene zu verankern statt an `z = 0`. Damit fällt `dz`
vollständig heraus, und **alles außer `ry ≥ 1.0°`** wäre abgeräumt.

**Aber:** Liegt der Deckflächen-Fit um δ daneben, steht δ an der Stelle von `dz`
und geht mit demselben Faktor `2·tan(α)` ein. Die Genauigkeit der Spaltbreite
hinge danach an der **Qualität der Deckflächen-Ebene** — deren Robustheit wird
damit sicherheitskritisch. Bei realen Scans sitzen dort Spritzer und
Reflexionen.

Der Gewinn ist trotzdem real: Die Deckfläche ist dicht besetzt und gut
konditioniert, anders als die `rx`-Rotation, die ICP schlecht auflöst.

**Schranken — die zweite ersetzt die erste, sie ergänzt sie nicht:**

- **jetzt** (z=0-Methode): für die 0.25-mm-Toleranz muss `dz < ~0.12 mm` bleiben.
  Synthetisch erfüllt (`dz < 0.04 mm`), bei realen Scans offen.
- **nach dem Fix**: `dz` fällt heraus, das Kriterium wird hinfällig. Maßgeblich
  ist dann der Fit-Fehler der Deckflächen-Ebene, mit derselben Schranke bei
  α = 45°; bei flacheren Nähten entsprechend lockerer.

### 5.5 Fixture ohne definierte Grundöffnung

Die V-Naht-Fixture in `tests/test_segmentation.py` läuft am Grund auf einen
Punkt zu (Spaltbreite ≈ 0). `gap_width_by_seam` ist dort deshalb nur auf
Achsen-Zuordnung geprüft, nicht quantitativ. TODO im Test vermerkt.

---

## 6. Ergebnis

| Größe | vorher | nachher |
|---|---|---|
| Fälle mit beiden Flanken | 0/61 | **61/61** |
| Reg-Residuum Median | 2.342 mm | **0.121 mm** |
| Fälle > 0.25 mm Residuum | 61/61 | **9/61** |
| Steigung Spaltbreite über `ty` | −1.04 | **+0.964** |
| max. Fehler `T_Y` | 3.7 mm | **0.078 mm** |

Fehlerbudget der verbleibenden 3.6 %: vollständig durch den
Verstärkungsmechanismus aus Abschnitt 4 erklärt, nicht durch die Messmethode.

---

## 7. Reproduktion

```bash
uv run python scripts/run_pipeline.py --batch
uv run python scripts/generate_synthetic_dataset.py
uv run python scripts/run_batch_subtraction.py \
    --scan-dir data/raw/synthetic_scans --source-type synthetic
```

Der erste Schritt braucht den CAD-API-Host
(`https://cimtt-ki.haw-kiel.de/cad-preprocessing-api/converter`) und legt den
rohen Cache unter `data/outputs/cad/<stem>/` an.

Auswertung: `notebooks/synthetic_validation.ipynb`,
Rohdaten `data/outputs/batch_summary.csv`.
