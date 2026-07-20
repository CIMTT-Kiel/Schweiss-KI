# Subtraktions-Pipeline (AP2.2 / AP2.3)

Stand: Zwischenstand nach Implementierung der Differenzanalyse und
werkstückweisen Registrierung, mit synthetischer Validierung auf
61 Testfällen.

## 1. Überblick

Die Subtraktions-Pipeline vergleicht einen realen 3D-Scan einer
Schweißbaugruppe mit dem idealen CAD-Modell und quantifiziert die
Abweichungen in vier abgestuften Analyse-Ebenen. Ausgangspunkt ist ein
vorverarbeiteter, segmentierter Scan im internen `WeldVolumeModel`.
Am Ende liefert die Pipeline pro Bauteil einen strukturierten
`SubtractionReport`, der die Registrierung und die Differenzanalyse
zusammenfasst.

Diese Doku beschreibt den aktuellen Implementierungsstand. Verwendetes
Testbauteil ist die V-Naht-Baugruppe mit 1,5 mm Sollspalt und 45°-Flanken.

## 2. Architektur

Die Subtraktion ist als Sub-Pipeline in die Hauptpipeline eingebettet und
läuft nach Preprocessing und Segmentierung des Scans. Sie besteht aus
zwei Phasen: Registrierung und Differenzanalyse.

```
Scan (WeldVolumeModel) + CAD (STEP)
        │
        ▼
[Registrierung]   Scan im CAD-Frame ausrichten
        │
        ▼
[Differenzanalyse] Abweichungen quantifizieren
        │
        ▼
SubtractionReport
```

Die Konfiguration erfolgt über `configs/pipeline.yaml` unter dem
`subtraction`-Block. Alle Steps sind unabhängig aktivierbar; die
Reihenfolge im YAML bestimmt die Ausführungsreihenfolge.

## 3. Registrierung

Zwei aufeinander aufbauende Steps richten den Scan im CAD-Koordinatensystem
aus.

**CoarsePCA** – grobe Vorausrichtung über Hauptachsen-Analyse. Nutzt nur
die Werkstück-Oberseiten (`anchor_labels: [0, 1, 2]`), um Verzerrungen
durch den Spaltbereich auszuschließen. Robust genug, um ICP-Konvergenz
zu ermöglichen, aber für einzelne Platten unbrauchbar (Symmetrie-Ambiguität).

**ICPFine** – Point-to-Plane ICP für die feine Registrierung. Ebenfalls
auf Anker-Labels beschränkt, `max_correspondence_distance = 1.0 mm`.

Der Registrierungs-Report enthält für jeden Step die Transformation,
Laufzeit, Residuum und Fitness. Auf synthetischen Testdaten erreicht die
Pipeline nach ICP durchgängig Registrierungs-Residuen unter 0,25 mm.

## 4. Differenzanalyse

Vier Steps liefern zunehmend spezifische Metriken.

### 4.1 PointDistance

Signierte Punkt-zu-CAD-Distanz für jeden Scan-Punkt:

$$d = (s - c) \cdot n_c$$

wobei `s` der Scan-Punkt, `c` der nächste CAD-Punkt (via KD-Tree) und
`n_c` die CAD-Normale an `c` ist. Vorzeichen aus dem Skalarprodukt mit
der Normalen (positiv: außerhalb, negativ: innerhalb).

Punkte mit `|d| > max_distance` werden aus den Aggregaten ausgeschlossen
(als Ausreißer behandelt, z. B. Sub-Gap-Artefakte). Aggregate:
`mean_signed`, `mean_abs`, `rms`, `max_abs`, `p95`, `in_tolerance_rate` –
global und pro Segmentierungs-Label.

Der Step liefert damit die im Projektplan geforderte allgemeine
Differenzbildanalyse.

### 4.2 VoxelDeviation

Räumliche Aufschlüsselung der Distanzen aus PointDistance in ein
regelmäßiges 3D-Voxel-Grid. Pro Zelle: Anzahl Punkte, mittlere signierte
Distanz, RMS, In-Toleranz-Anteil. Voxel mit weniger als
`min_points_per_voxel` Punkten werden verworfen.

Ergebnis ist eine ortsaufgelöste Beschreibung der Abweichung, die
lokale Defekte sichtbar macht, welche bei einer globalen Mittelung
untergehen würden.

### 4.3 ComponentRegistration

Werkstückweise Registrierung mit Ableitung der relativen Lage.

Der Scan wird räumlich in die beiden Werkstücke geteilt (Trennebene
`split_axis: 1`, `split_value: 0.0`; Werkstück A liegt bei Y ≥ 0,
Werkstück B bei Y < 0). Jedes Werkstück wird einzeln gegen sein CAD-Pendant
registriert, wobei nur ICPFine läuft – CoarsePCA würde an der uniformen
Einzelplatte an einer Symmetrie-Ambiguität scheitern.

Aus den beiden Transformationen `T_A` und `T_B` wird die relative
Transformation gebildet:

$$T_{rel} = T_B \cdot T_A^{-1}$$

`T_rel` beschreibt, wie Werkstück B gegenüber Werkstück A im realen
Bauteil verschoben und verkippt ist – invariant gegen die Lage des
Gesamtbauteils in der Scanner-Kammer.

**Zerlegung um den Werkstück-Schwerpunkt.** `T_rel` würde standardmäßig
um den Ursprung zerlegt. Bei Rotationen um einen Punkt außerhalb des
Werkstück-Schwerpunkts entsteht dabei ein Translations-Anteil, der die
Interpretation verwässert – eine reine Rotation würde als kombinierte
Rotation-plus-Translation erscheinen. Um Rotation und Translation sauber
zu trennen, wird der Translations-Anteil um die rotations-induzierte
Schwerpunktsverschiebung bereinigt:

$$t_{pure} = t_{raw} - (I - R_{rel}) \cdot c_B$$

`c_B` ist das Zentrum des Werkstück-B-Targets im CAD-Frame. Damit gilt:
bei reiner Rotation ist `t_pure ≈ 0`, bei reiner Translation ist
`t_pure = t_raw`.

Der Report enthält sechs Freiheitsgrade als Ergebnis: Translation
`(Δx, Δy, Δz)` in mm und Rotation `(rx, ry, rz)` als Euler-Winkel
in Grad (XYZ-Konvention). Zusätzlich werden `translation_raw_mm`,
`rotation_center_mm` sowie die vollständigen Transformationsmatrizen
gespeichert.

### 4.4 GapProfile

Wurzelspalt-Verlauf entlang der Naht-Längsrichtung, V-Naht-spezifisch.
Der Scan wird in N Bins entlang der X-Achse geteilt; pro Bin wird über
lineare Extrapolation der Flanken auf Z = 0 die effektive Spaltbreite
berechnet.

Dieser Step liefert die Bauteil-spezifische Detail-Metrik, ist aber nicht
verallgemeinerbar auf andere Nahtgeometrien.

## 5. Validierung

Die Pipeline wurde gegen einen synthetischen Datensatz mit bekannter
Ground Truth validiert.

### 5.1 Synthetischer Datensatz

61 Testfälle, generiert aus dem CAD-Modell durch Transformation von
Werkstück B mit definierten Translation- und Rotationswerten. Werkstück A
bleibt stationär. Die eingebrachte Transformation ist damit exakt die
relative Lage-Abweichung, die die Pipeline zurückgeben soll.

**Kategorien:**
- Translation in X, Y, Z (jeweils einzeln) – 21 Fälle
- Rotation um X, Y, Z (jeweils einzeln) – 18 Fälle
- Translation-Kombinationen – 5 Fälle
- Rotation-Kombinationen – 5 Fälle
- 6-DOF-Kombinationen – 12 Fälle

**Konvention der Metadaten:** `+ty` in der CSV bedeutet, dass sich der
Spalt öffnet (Werkstücke gehen auseinander). Der Generator invertiert
`ty` intern beim Anwenden auf B, da B im negativen Y-Bereich liegt.

Der Datensatz und der Generator sind unter `scripts/generate_synthetic_dataset.py`
implementiert; die Metadaten liegen in `data/raw/synthetic_scans/synthetic_metadata.csv`.

### 5.2 Prüfmethodik

Zwei Notebooks decken die Validierung ab:

`synthetic_inspection.ipynb` – Sanity-Check der Ground Truth: rekonstruiert
die erwartete synthetische Wolke aus Referenz-A und transformiertem
Referenz-B, vergleicht sie punktweise mit dem tatsächlichen Generator-Output.
Die maximale Distanz muss ≈ 0 sein, dann arbeitet der Generator korrekt.

`synthetic_validation.ipynb` – Vergleich der Pipeline-Ergebnisse mit den
Ground-Truth-Werten. Beinhaltet Scatter-Plots pro Freiheitsgrad,
Fehler-Statistik pro Kategorie und Deep-Dives für auffällige Muster.

Wichtig für die Interpretation ist die Vorzeichen-Konvention: die
Pipeline gibt `T_rel = T_B · T_A⁻¹` zurück, was die Inverse der
eingebrachten Transformation ist. Zusätzlich wird `ty` im Generator
invertiert (User-Konvention). Die daraus resultierenden Vorzeichen der
erwarteten gemessenen Werte sind im Validierungs-Notebook als
`SIGN`-Dictionary explizit hinterlegt.

## 6. Ergebnisse

Für 4 der 6 Freiheitsgrade liefert die Pipeline im validierten Bereich
Werte innerhalb der Toleranz. Fehler-Statistik über alle 61 Testfälle:

| DOF     | mean(|err|) | p95(|err|) | max(|err|) |
|---------|-------------|------------|------------|
| tx_mm   | 0.286       | 0.605      | 0.605      |
| ty_mm   | 0.062       | 0.203      | 1.103      |
| tz_mm   | 0.037       | 0.163      | 0.208      |
| rx_deg  | 0.276       | 0.418      | 5.870      |
| ry_deg  | 0.033       | 0.239      | 0.418      |
| rz_deg  | 0.021       | 0.117      | 0.242      |

Die Y- und Z-Translation sowie Y- und Z-Rotation werden präzise
rekonstruiert. Die X-Translation und die X-Rotation zeigen bei
Kombinationsfällen systematische Fehler.

Die Bereinigung um den Werkstück-Schwerpunkt hat den Rotation-Translation-Effekt
weitgehend eliminiert. Reine Rotationen um Y ergeben nach der Bereinigung
Δz ≈ 0 (vorher bis zu 3,48 mm).

## 7. Bekannte Einschränkungen

**X-Translation:** Die Pipeline erkennt Verschiebungen entlang der
Naht-Längsachse unzuverlässig. Bei einer Ground Truth von +5 mm werden
nur -4,4 mm gemessen. Ursache ist die geometrische Uniformität des
Werkstücks entlang der X-Achse: ICP hat keine ausreichenden Merkmale
für die Registrierung in dieser Richtung. In der Praxis ist die
X-Verschiebung als isolierter Fertigungsfehler unwahrscheinlich, aber
in Kombinationsszenarien kann sie zu Fehlern beitragen. Ein Fix (z. B.
über Feature-basierte X-Ankerung oder erweiterte Anker-Auswahl) ist der
nächste geplante Schritt.

**Kombinierte 6-DOF-Fehler:** Bei stark überlagerten Fehlern (mehrere
DOFs gleichzeitig auf großen Werten) versagt die Pipeline zunehmend.
`C_TR_11` mit tx = 2, ty = -1, tz = 0,5, rx = 1°, ry = 0,5°, rz = 0,5°
ergibt einen rx-Fehler von 5,87°. Das ist ein grundsätzliches
ICP-Konvergenz-Problem, das erst bei sehr großen kombinierten
Fehlerbildern auftritt.

**V-Naht-Spezifik:** Der GapProfile-Step, die Segmentierungs-Labels und
der `split_axis: 1`-Split in ComponentRegistration sind auf die V-Naht-Geometrie
zugeschnitten. Für andere Nahttypen ist eine Erweiterung erforderlich.

**Realdaten-Validierung offen:** Die aktuelle Validierung basiert
ausschließlich auf synthetischen Daten. Für echte Scans mit
Preprocessing-Artefakten (Ausreißer, Rauschen, Scan-Lücken) fehlen
kontrollierte Testszenarien mit Ground Truth. Die vorhandenen fünf
CMM-Scans zeigen plausibles Verhalten, aber ohne Ground Truth ist eine
quantitative Aussage über die Genauigkeit auf Realdaten nicht möglich.

## 8. Ausblick

Konkret geplant:
- X-Translations-Problem angehen: bessere Anker-Auswahl oder
  X-Vorregistrierung über Feature-Punkte
- Realdaten-Validierung strukturieren: entweder mit vermessenen
  Referenz-Bauteilen oder mit Cross-Validation gegen Handmessungen
- Feature-Vektor-Aufbereitung (Phase 4 aus dem Konzept) als
  Schnittstelle zu AP3

Mittelfristig, abhängig von der Datenlage:
- Generalisierung der Segmentierung und Split-Achse auf beliebige
  Nahttypen
- Erweiterung um komplexere Bauteilgeometrien, sobald Heidenbluth
  Referenz-Bauteile bereitstellt

## 9. Dateien und Struktur

```
src/schweiss_ki/subtraction/
├── base.py                       # abstrakte Basisklassen
├── reports.py                    # SubtractionReport, DeviationData, ...
├── plots.py                      # Diagnose-Plots
├── registration/
│   ├── pipeline.py               # RegistrationPipeline
│   ├── coarse_pca.py             # CoarsePCA
│   └── icp_fine.py               # ICPFine
└── deviation/
    ├── pipeline.py               # DeviationPipeline
    ├── point_distance.py         # PointDistance
    ├── voxel_deviation.py        # VoxelDeviation
    ├── component_registration.py # ComponentRegistration
    └── gap_profile.py            # GapProfile

scripts/
├── run_batch_subtraction.py      # Batch-Lauf über ein Scan-Verzeichnis
└── generate_synthetic_dataset.py # Erzeugung der 61 synthetischen Bauteile

notebooks/
├── validation_dashboard.ipynb    # Diagnose-Notebook für einzelne Bauteile
├── synthetic_inspection.ipynb    # Prüfung der synthetischen Ground Truth
└── synthetic_validation.ipynb    # Ground Truth vs. Pipeline-Ergebnisse

configs/pipeline.yaml             # zentrale Konfiguration
data/raw/synthetic_scans/         # synthetische PLYs + Metadaten-CSV
data/outputs/                     # Pipeline-Output pro Bauteil
```
