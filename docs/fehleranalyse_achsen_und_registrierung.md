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

### 3.5 Ein Deckflächen-Fit für zwei Werkstücke (`139a528`)

`background_remover` fittete **eine** Ebene über beide Deckflächen. Sobald die
Werkstücke gegeneinander verkippt sind, findet RANSAC damit die dominante Hälfte
und behandelt die andere als Ausreißer:

| `R_Y_+01.000deg` | Punkte |
|---|---|
| Deckflächenpunkte gesamt | 435.483 |
| in der gefundenen Ebene | 251.284 |
| durchgefallen | **184.199 (42 %)** |

Die durchgefallenen Punkte bekamen Label 0 erst am Pipeline-Ende über
`fill_unlabeled_with_background` — also über das Sicherheitsnetz statt über eine
Klassifikation.

**Behebung:** zwei getrennte Fits, aufgeteilt am Vorzeichen der `gap_axis`
(`split_gap_axis`, Default 1 = Y, passend zu `component_registration`). Je Seite
werden Ebene, Neigung und Inlier-Zahl ausgewiesen, dazu die relative Verkippung
beider Ebenen als eigenes Merkmal. `split_gap_axis=None` stellt das alte
Verhalten wieder her.

#### Der Nebeneffekt: Nichtdeterminismus zwischen Läufen

Zwei Läufe desselben Datensatzes lieferten nicht exakt dieselben Zahlen — anfangs
0.013 mm Drift in 14 von 61 Fällen. Die Diagnose lief zunächst in die falsche
Richtung: RANSAC ist zufällig, also fehlt ein Seed. Ein Seed reduzierte den Drift
(auf 0.004 mm), beseitigte ihn aber nicht. Auch eine Fixierung der
OpenMP-Threadzahl wurde geprüft und wieder verworfen — im realistischen Aufbau
war der ungedrosselte Default über 3 Prozesse × 30 Läufe selbst bitgleich, die
Maßnahme hätte also nichts belegt.

**Die eigentliche Ursache war der Ein-Ebenen-Fit oben.** Die Streuung kam nicht
aus RANSACs Zufallsstichprobe, sondern aus der **Mehrdeutigkeit**: Bei zwei
konkurrierenden Deckflächen musste der gemeinsame Fit zwischen ihnen wählen, und
die Wahl fiel je nach Stichprobe anders aus. Je Seite getrennt gefittet gibt es
nur eine Ebene — und die findet RANSAC stabil.

Gemessen an `R_Y_+01.000deg`, 12 Läufe mit *wechselnden* Seeds:

| | Ergebnisse | Spanne |
|---|---|---|
| ein gemeinsamer Fit | 12 verschiedene | 0.016 mm |
| zwei getrennte Fits | **1** | **0.000 mm** |

`C_TR_08` und `T_X_+00.100mm` waren in beiden Varianten stabil — der Effekt trat
ausschließlich bei verkippten Werkstücken auf, also genau dort, wo zwei
Deckflächen konkurrierten.

**Bestätigt über den vollen Batch-Pfad**, zwei komplette Läufe mit gleichem Seed:

| | identische Fälle | max. Abweichung |
|---|---|---|
| vor dem Fix | 47/61 | 0.013 mm |
| nach dem Fix | **61/61** | **0.000000 mm** |

Bitgleich in jeder Spalte: Spaltwerte, Registrierungs-Residuen,
`inlier_ratio`, Bin-Zahlen.

**Konsequenz für die Reproduzierbarkeit:** Das Ergebnis hängt nicht mehr am Seed.
Die Seed-Infrastruktur (`core/reproducibility.py`, `random_seed` in der Config,
`--seed`, effektiver Seed je Modell im Report) bleibt, weil sie protokolliert,
womit ein Ergebnis entstand — für den Determinismus ist sie aber nicht mehr
tragend. Abgesichert durch `tests/test_reproducibility.py`, dessen Gegenprobe
explizit den alten Ein-Ebenen-Fit prüft und damit belegt, dass die Mehrdeutigkeit
die Ursache war.

**Lehre für die Diagnose:** Von „zwei Läufe, andere Zahlen" wurde direkt auf die
naheliegende Erklärung geschlossen (RANSAC ist zufällig → Seed fehlt), statt zu
prüfen, ob die Ursache eine Modellierungsebene höher liegt. Fünf Messrunden für
einen Effekt bei 5 % der Toleranz; der später ohnehin geplante Aufräum-Commit
löste ihn nebenbei.

---

## 4. Der Verstärkungsmechanismus der Spaltmessung

> **Behoben** (`4a8bbae`) — die Auswertungshöhe ist jetzt an der
> Deckflächen-Ebene verankert. Der Abschnitt beschreibt den Mechanismus, weil er
> für andere Öffnungswinkel und für die realen Scans relevant bleibt; die
> verbleibende Abhängigkeit steht in 5.4.

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

### 4.1 Behebung: Verankerung an der Deckflächen-Ebene (`4a8bbae`)

Die Auswertungshöhe zielt nicht mehr auf `vertical_axis = 0`, sondern auf die
aus dem Scan gefittete Deckflächen-Ebene des Referenz-Werkstücks. Damit fällt
`dz` strukturell heraus, statt verdoppelt einzugehen.

Registrierungseinfluss, isoliert gemessen (registriert gegen unregistriert):

| Kategorie | n | max\|e\| vorher | max\|e\| nachher |
|---|---|---|---|
| `rotation_x` | 10 | 0.9421 | **0.0004** |
| `rotation_y` | 4 | 0.8488 | **0.0037** |
| `translation_z` | 4 | 0.9871 | **0.0012** |
| `rotation_combo` | 5 | 1.2979 | **0.0078** |
| **alle** | **61** | **1.2979** | **0.0078** |
| RMS | | 0.4154 | **0.0017** |
| innerhalb 0.25 mm | | 63.9 % | **100 %** |

Auch `ry ≥ 1.0°` ist mit abgeräumt, obwohl dort die Registrierung qualitativ
einbricht (5.1): Der Einbruch ist eine Starrkörper-Fehlstellung, und gegen die
ist die verankerte Messung invariant — unabhängig von ihrer Größe.

Weitere Konsequenzen der Umstellung:

- Die Spaltbreite ist als Funktion der **Tiefe unter der Referenz-Deckfläche**
  parametrisiert, nicht als Einzelwert. Ausgegeben werden je Flanke und Bin
  Achsenabschnitt, Steigung und Gütemaß — daraus lassen sich Flankenwinkel
  (gemessen, nicht vorausgesetzt), Asymmetrie zwischen den Flanken und der
  Wurzelspalt ableiten.
- Der Wurzelspalt ist ein **Messwert** an der tiefsten beidseitig besetzten
  Tiefe, keine Extrapolation. Verbleibender systematischer Offset: −0.019 mm
  aus dem P95-Schnitt bei `d_root`, konstant und ohne Streuung.
- Die Lage des Gegen-Werkstücks relativ zur Referenz (Höhenversatz, Verkippung)
  wird als eigenes Qualitätsmerkmal ausgegeben — es bildet Kantenversatz ab und
  wird **nicht** in die Höhenreferenz eingemittelt.
- Abgesichert durch einen `dz`-Invarianztest (Toleranz 1e-6) mit Gegenprobe,
  dass die alte Methode den `−2·dz`-Effekt zeigt. Mutationstest: Verankerung
  ausgehebelt → 7 Tests fallen.

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

**Folge für die Spaltmessung: keine.** Der Einbruch ist eine
Starrkörper-Fehlstellung, und gegen die ist die verankerte Messung invariant
(4.1) — unabhängig von ihrer Größe. Der Registrierungseinfluss liegt auch bei
`rotation_y` bei max. 0.0037 mm. Betroffen bleibt allein, wie gut Scan und CAD
zueinander liegen (Reg-Residuum 0.482 statt ~0.12 mm), was für
`point_distance` und `voxel_deviation` relevant ist, nicht für den Wurzelspalt.

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

### 5.4 Die Abhängigkeit ist verlagert, nicht beseitigt

Der Fix aus Abschnitt 4.1 verankert die Auswertungshöhe an der
RANSAC-Deckflächenebene. `dz` fällt damit heraus — **an seine Stelle tritt aber
der Fit-Fehler δ dieser Ebene, mit demselben Faktor `2·tan(α)`.**

Die Genauigkeit der Spaltbreite hängt jetzt an der **Qualität der
Deckflächen-Ebene**; deren Robustheit ist damit sicherheitskritisch. Bei
synthetischen Daten unkritisch (`inlier_ratio` ≥ 0.993, `rms` ≈ 0.013 mm), bei
realen Scans sitzen dort aber Spritzer und Reflexionen.

Abgesichert ist das durch:
- **RANSAC statt Least-Squares** für die Deckfläche — ein LSQ-Fit würde mit den
  Ausreißern mitwandern
- **`inlier_ratio` und `rms` als Gütemaße** im Report
- **ein Gate**: unterschreitet `inlier_ratio` den Schwellwert, wird die
  Verankerung verweigert und *kein* Spaltwert ausgegeben, statt einen falschen
  Bezug zu liefern
- Tests mit künstlichen Spritzern (5 / 15 / 30 % Störanteil), die belegen, dass
  RANSAC den Fit hält und `inlier_ratio` monoton mit der Störung fällt

**Schranke:** Für die 0.25-mm-Toleranz muss δ unter ~0.12 mm bleiben (bei
α = 45°; bei flacheren Nähten entsprechend lockerer). Das ersetzt die frühere
Schranke `dz < 0.12 mm` der z=0-Methode — es ergänzt sie nicht.

### 5.5 Fixture ohne definierte Grundöffnung

Die V-Naht-Fixture in `tests/test_segmentation.py` läuft am Grund auf einen
Punkt zu (Spaltbreite ≈ 0). `gap_width_by_seam` ist dort deshalb nur auf
Achsen-Zuordnung geprüft, nicht quantitativ. TODO im Test vermerkt.

Für `GapProfile` existiert in `tests/test_gap_profile.py` inzwischen eine eigene
Fixture **mit** definierter Wurzelöffnung — dort ist der gemessene Spalt gegen
einen bekannten Wert prüfbar.

### 5.6 `C_TR_08` liefert 15/20 Bins

Der einzige Fall, in dem das NaN-Gate Bins verwirft. Es arbeitet dabei wie
vorgesehen: statt einen Geradenfit über ein zu dünnes Tiefenband zu legen, wird
der Bin verworfen.

Die Ursache liegt in der **Flanken-Abdeckung**, nicht in der Mindestbesetzung:
Nachgemessen haben 0 von 20 Bins weniger als 10 Punkte auf einer Flanke.
Sichtbar wird es im Querschnittsplot — Flanke A bricht bei z ≈ 1.6 ab, während B
bis z ≈ −1.0 reicht; wo `[d_min, P95]` nicht genug Spreizung hat, greift
`min_depth_span`.

Die Vermutung, der Zwei-Ebenen-Fit (3.5) würde das entspannen, hat sich **nicht
bestätigt**: vorher wie nachher 60 von 61 Fällen mit 20/20. Deckflächen- und
Flankenabdeckung hängen hier nicht zusammen.

`C_TR_08` ist der Fall mit dem schlechtesten Reg-Residuum (0.693 mm) — dass
gerade dort die Flankenabdeckung leidet, ist konsistent.

---

## 6. Ergebnis

| Größe | vorher | nachher |
|---|---|---|
| Fälle mit beiden Flanken | 0/61 | **61/61** |
| Reg-Residuum Median | 2.342 mm | **0.121 mm** |
| Fälle > 0.25 mm Residuum | 61/61 | **9/61** |
| Steigung Spaltbreite über `ty` | −1.04 | **+1.000** |
| Registrierungseinfluss, max über alle 61 | 1.298 mm | **0.008 mm** |
| Registrierungseinfluss, RMS | 0.415 mm | **0.002 mm** |
| innerhalb 0.25-mm-Toleranz | 63.9 % | **100 %** |
| Deckfläche über Sicherheitsnetz (`R_Y_+1.0°`) | 42.5 % | **0 %** |
| `inlier_ratio` Referenzebene, Minimum | 0.992 | **1.000** |

Der verbleibende systematische Offset ist der P95-Schnitt bei `d_root`:
**−0.019 mm**, konstant und ohne Streuung über die `T_Y`-Serie. Er ist über
`flank_depth_max_quantile` justierbar; P95 wurde beibehalten, weil das echte
Maximum ausreißerempfindlich wäre.

Die vier großen Fehlerquellen — Achsen-Konvention, `coarse_pca`,
Unterseiten-Kontamination und die z=0-Auswertungshöhe — sind behoben. Was
bleibt, steht in Abschnitt 5 und liegt durchweg unterhalb der Toleranz.

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
rohen Cache unter `data/outputs/cad/<stem>/` an. Der Generator liest **diesen**
Cache, nicht die vorverarbeitete Wolke unter `data/outputs/<stem>/` (siehe 3.3);
`load_cad()` bricht bei Verwechslung mit einer Meldung ab.

Auswertung: `notebooks/synthetic_validation.ipynb`,
Rohdaten `data/outputs/batch_summary.csv`.

**Reproduzierbarkeit.** Zwei Läufe mit gleichem `random_seed` liefern identische
Werte (siehe 3.5). Der effektive Seed steht je Modell in
`subtraction_report.json`; über `--seed` lässt er sich überschreiben, etwa um die
Streuung gegen die RANSAC-Wahl zu quantifizieren.

**Visuelle Kontrolle.** `plot_cross_section(model, x_min, x_max)` zeichnet die im
Report gespeicherten Fits — Referenzebene, beide Flankengeraden über ihrem
tatsächlichen Tiefenband, Fitband-Obergrenze und Wurzeltiefe. Bewusst keine neu
gerechneten: ein zweiter Fit im Plot würde Abweichungen zwischen Bild und
Messwert verstecken.
