"""
Tests für die Reproduzierbarkeit (Seeding der RANSAC-Schritte).

Ausgangslage: background_remover und flank_segmenter nutzen Open3Ds
segment_plane, also RANSAC mit Zufallsstichprobe. Ohne Seed wechselten auf
einer realen Wolke bis zu 10.979 von 504.200 Punkten (2.2 %) zwischen zwei
Läufen das Label, was sich als ~0.013 mm Streuung in der Spaltbreite
niederschlug. Für die 0.25-mm-Toleranz unkritisch, als Methodik aber nicht
haltbar – und vor allem: ohne feste Werte ist kein Referenzvergleich
zwischen zwei Läufen definierbar.

Aufgeloest wurde das NICHT durch den Seed, sondern durch den
Zwei-Ebenen-Fit in background_remover: die Streuung kam aus der
Mehrdeutigkeit zwischen zwei konkurrierenden Deckflaechen, nicht aus
RANSACs Zufallsstichprobe an sich. Fittet man beide Werkstueckseiten
getrennt, gibt es je Seite nur eine Ebene – und die findet RANSAC stabil.

Gemessen an R_Y_+01.000deg ueber 12 Laeufe mit wechselnden Seeds:
    ein gemeinsamer Fit : 12 verschiedene Ergebnisse, Spanne 0.016 mm
    zwei getrennte Fits :  1 Ergebnis, Spanne 0.000 mm

Die Seed-Infrastruktur bleibt sinnvoll (protokolliert, womit ein Ergebnis
entstand), ist fuer den Determinismus aber nicht mehr der tragende Teil.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from schweiss_ki.core.reproducibility import (
    DEFAULT_SEED,
    derive_seed,
    seed_everything,
    seed_for_model,
)
from schweiss_ki.segmentation import SegmentationPipeline, NAME_TO_ID
from schweiss_ki.segmentation import BackgroundRemover, FlankSegmenter, GapClassifier

# 15 statt 5: die zu erkennende Varianz ist flakig (der alte Ein-Ebenen-Fit
# lieferte bei gleichem Seed mal 1, mal 2 verschiedene Ergebnisse). Mit 5
# Wiederholungen schlug der Mutationstest nur unzuverlaessig an.
#
# Die urspruengliche Vorgabe waren 30 Laeufe ueber mehrere Prozesse. In der
# Testsuite ist das nicht praktikabel - pytest laeuft in einem Prozess, und
# 30 Segmentierungen je Test wuerden die Suite deutlich verlangsamen. 15
# Laeufe in einem Prozess erkennen die Regression zuverlaessig; die
# prozessuebergreifende Bestaetigung ist einmalig manuell erfolgt und in
# docs/fehleranalyse_achsen_und_registrierung.md festgehalten.
N_RUNS = 15


@pytest.fixture
def v_seam_pcd():
    """V-Naht mit VERKIPPTEM Gegenstueck – RANSAC muss zwischen zwei
    konkurrierenden Deckflaechen-Ebenen waehlen.

    Eine plane, saubere Naht genuegt hier NICHT: dort findet RANSAC
    unabhaengig vom Seed immer dieselbe Ebene, und der Determinismus-Test
    waere vakuum-gruen. Genau dieser Fall – ein gegen die Referenz
    verkipptes Werkstueck – erzeugte auf den realen Daten die groesste
    Streuung (R_Y, C_TR_*), weil dort 42 % der Deckflaeche nicht mehr in die
    dominante Ebene passen.
    """
    from test_gap_profile import make_v_seam
    pcd, _ = make_v_seam(noise=0.03, n_top=8000, n_flank=4000)

    # Negative gap-Seite um die Naht-Laengsachse kippen
    pts = np.asarray(pcd.points).copy()
    side = pts[:, 1] < 0
    angle = np.deg2rad(1.5)
    c, s = np.cos(angle), np.sin(angle)
    y, z = pts[side, 1].copy(), pts[side, 2].copy()
    pts[side, 1] = c * y - s * z
    pts[side, 2] = s * y + c * z
    pcd.points = o3d.utility.Vector3dVector(pts)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 100.0]))
    return pcd


def _segment(pcd, split_gap_axis: int | None = 1):
    """Segmentierung; split_gap_axis=None entspricht dem alten Ein-Ebenen-Fit."""
    pipeline = SegmentationPipeline([
        BackgroundRemover(split_gap_axis=split_gap_axis),
        FlankSegmenter(expected_flank_angle_deg=45.0),
        GapClassifier(),
    ])
    labels, _ = pipeline.process(pcd)
    return labels


class TestSeedDerivation:
    def test_derived_seed_is_stable(self):
        """Gleicher Basis-Seed und gleiche ID -> gleicher abgeleiteter Seed.

        Muss auch ueber Prozessgrenzen gelten. Pythons eingebautes hash() ist
        fuer Strings via PYTHONHASHSEED gesalzen und waere hier untauglich –
        deshalb blake2b.
        """
        assert derive_seed(0, "C_TR_08") == derive_seed(0, "C_TR_08")
        # Bekannter Wert, faellt auf, falls jemand das Verfahren tauscht
        assert derive_seed(0, "C_TR_08") == 1188428860

    def test_different_models_get_different_seeds(self):
        ids = ["T_X_+00.100mm", "T_Y_+01.500mm", "C_TR_08", "R_Y_+01.000deg"]
        seeds = {derive_seed(0, m) for m in ids}
        assert len(seeds) == len(ids)

    def test_different_base_seeds_are_independent(self):
        """Basis-Seeds duerfen keine ueberlappenden Sequenzen erzeugen.

        Bei base + index waere Basis 0/Modell 5 identisch mit Basis 5/Modell 0.
        Fuer eine Streuungsanalyse ueber mehrere Basis-Seeds muessen die
        Laeufe unabhaengig sein.
        """
        a = {derive_seed(0, f"m{i}") for i in range(20)}
        b = {derive_seed(1, f"m{i}") for i in range(20)}
        assert not (a & b)

    def test_seed_for_model_returns_effective_seed(self):
        eff = seed_for_model(0, "C_TR_08")
        assert eff == derive_seed(0, "C_TR_08")


class TestSegmentationDeterminism:
    def test_single_plane_fit_varies(self, v_seam_pcd):
        """Gegenprobe: der ALTE Ein-Ebenen-Fit streut auf dieser Fixture.

        Belegt zweierlei:
        1. Die Fixture erzeugt ueberhaupt RANSAC-Varianz – ohne das waere der
           Determinismus-Test unten aussagelos.
        2. Die Varianz kam aus der MEHRDEUTIGKEIT, nicht aus RANSACs
           Zufallsstichprobe an sich: bei zwei gegeneinander verkippten
           Deckflaechen musste ein gemeinsamer Fit zwischen ihnen waehlen, und
           die Wahl fiel je nach Stichprobe anders aus.
        """
        runs = []
        for i in range(N_RUNS):
            seed_everything(1000 + i)
            runs.append(_segment(v_seam_pcd, split_gap_axis=None))
        assert any((r != runs[0]).any() for r in runs[1:]), (
            "Fixture erzeugt keine RANSAC-Varianz – Determinismus-Test waere aussagelos"
        )

    def test_seeded_segmentation_is_bit_identical(self, v_seam_pcd):
        """Kernforderung: gleicher Seed -> bitgleiche Labels ueber N Laeufe."""
        runs = []
        for _ in range(N_RUNS):
            seed_for_model(DEFAULT_SEED, "determinismus-test")
            runs.append(_segment(v_seam_pcd))

        for i, r in enumerate(runs[1:], start=2):
            n_diff = int((r != runs[0]).sum())
            assert n_diff == 0, (
                f"Lauf {i} weicht in {n_diff} von {len(r)} Labels ab – "
                f"es bleibt eine ungeseedete RNG-Quelle"
            )

    def test_label_counts_stable(self, v_seam_pcd):
        """Zusaetzliche Sicht auf dieselbe Forderung, mit lesbarer Diagnose."""
        counts = []
        for _ in range(N_RUNS):
            seed_for_model(DEFAULT_SEED, "determinismus-test")
            lbl = _segment(v_seam_pcd)
            counts.append(tuple(int((lbl == k).sum()) for k in (0, 1, 2, 3, 4)))
        assert len(set(counts)) == 1, f"Label-Verteilung schwankt: {set(counts)}"


class TestMeasurementDeterminism:
    def test_gap_measurement_bit_identical(self, v_seam_pcd):
        """Der Messwert selbst – das, was am Ende zaehlt – ist reproduzierbar."""
        from schweiss_ki.subtraction.deviation.gap_profile import GapProfile
        from schweiss_ki.subtraction.reports import DeviationData

        step = GapProfile(seam_axis=0, gap_axis=1, vertical_axis=2)
        values = []
        for _ in range(N_RUNS):
            seed_for_model(DEFAULT_SEED, "determinismus-test")
            labels = _segment(v_seam_pcd)
            data = DeviationData()
            art = step._apply(v_seam_pcd, v_seam_pcd, data, labels, None)
            values.append(art.get("gap_root_mean_mm"))

        assert all(v == values[0] for v in values), (
            f"Spaltbreite schwankt ueber {N_RUNS} Laeufe: {values}"
        )

    def test_result_is_seed_independent(self, v_seam_pcd):
        """Mit dem Zwei-Ebenen-Fit haengt das Ergebnis nicht mehr am Seed.

        Die staerkere Aussage gegenueber blossem Determinismus: es gibt keine
        RANSAC-Wahl mehr zu treffen, weil je Werkstueckseite nur eine Ebene
        existiert. Der Seed ist damit fuer die Reproduzierbarkeit nicht mehr
        tragend – er bleibt nur als Protokoll, womit ein Ergebnis entstand.

        Faellt dieser Test, ist wieder eine Mehrdeutigkeit im Spiel und die
        Streuung ueber Seeds waere neu zu quantifizieren.
        """
        from schweiss_ki.subtraction.deviation.gap_profile import GapProfile
        from schweiss_ki.subtraction.reports import DeviationData

        step = GapProfile(seam_axis=0, gap_axis=1, vertical_axis=2)
        values = []
        for base in range(N_RUNS):
            seed_for_model(base, "streuung")
            labels = _segment(v_seam_pcd)
            data = DeviationData()
            art = step._apply(v_seam_pcd, v_seam_pcd, data, labels, None)
            values.append(art.get("gap_root_mean_mm"))

        assert all(v is not None for v in values)
        spread = max(values) - min(values)
        assert spread == 0.0, (
            f"Ergebnis haengt ueber {N_RUNS} Basis-Seeds noch am Seed "
            f"(Spanne {spread:.6f} mm) – es gibt wieder eine Mehrdeutigkeit"
        )
