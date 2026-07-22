#!/usr/bin/env python3
"""Test CAD API: STEP → PLY Konvertierung mit target_point_spacing (v1.0.8)."""

from pathlib import Path
from client.core import CADConverterClient
import time
from schweiss_ki.core.console import force_utf8_output

# ── Konfiguration ─────────────────────────────────────────────────────────
STEP_FILE       = Path("data/raw/step_files/Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt.STEP")
OUTPUT_DIR      = Path("data/processed/Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt")
TARGET_SPACING  = 0.1     # mm – mittlerer Punktabstand. None = API-Default
# ──────────────────────────────────────────────────────────────────────────


def test_conversion():
    client = CADConverterClient()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_spacing{TARGET_SPACING}" if TARGET_SPACING else "_default"
    output_file = OUTPUT_DIR / f"pointcloud{suffix}.ply"

    print("=" * 70)
    print("CAD API Test: STEP → PLY Konvertierung")
    print("=" * 70)
    print(f"\n📁 Input:           {STEP_FILE}")
    print(f"📁 Output:          {output_file}")
    print(f"🎯 target_spacing:  {TARGET_SPACING if TARGET_SPACING else 'API-Default'}")

    if not STEP_FILE.exists():
        print(f"\n❌ STEP-Datei nicht gefunden: {STEP_FILE}")
        return False

    # ── point_count aus Bauteil-Oberfläche ableiten ───────────────────────
    kwargs = {}
    if TARGET_SPACING is not None:
        try:
            print(f"\n🔍 Analysiere CAD-Oberfläche...")
            t = time.time()
            analysis = client.analyse_cad(str(STEP_FILE))
            objects = analysis["objects"]
            total_area = sum(obj["surface_area"] for obj in objects)
            point_count = max(1, int(total_area / TARGET_SPACING ** 2))
            kwargs["point_count"] = point_count
            print(f"   Dauer:              {time.time() - t:.2f} s")
            print(f"   Objekte:            {len(objects)}")
            print(f"   Bauteil-Oberfläche: {total_area:,.1f} mm²")
            print(f"   → point_count:      {point_count:,}")
        except Exception as e:
            print(f"\n⚠️  analyse_cad fehlgeschlagen ({type(e).__name__}: {e})")
            print(f"   Fallback auf API-Default-Sampling.")

    # ── Konvertierung ─────────────────────────────────────────────────────
    print(f"\n🔄 Starte Konvertierung...")
    start_time = time.time()

    try:
        path = client.convert_to_ply(str(STEP_FILE), str(output_file), **kwargs)
        elapsed = time.time() - start_time

        print(f"\n✅ Konvertierung erfolgreich!")
        print(f"   PLY gespeichert: {path.absolute()}")
        print(f"   Dauer:           {elapsed:.2f} s")

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\n📊 Output File:")
        print(f"   Größe:   {size_mb:.2f} MB")

        # ── Stats ─────────────────────────────────────────────────────────
        try:
            import open3d as o3d
            import numpy as np

            pcd = o3d.io.read_point_cloud(str(path))
            n = len(pcd.points)

            print(f"\n📐 Point Cloud Stats:")
            print(f"   Punkte:        {n:,}")
            if "point_count" in kwargs:
                delta = n - kwargs["point_count"]
                print(f"   Angefordert:   {kwargs['point_count']:,}  (Δ = {delta:+,})")
            print(f"   Hat Normalen:  {pcd.has_normals()}")
            print(f"   Hat Farben:    {pcd.has_colors()}")

            bbox = pcd.get_axis_aligned_bounding_box()
            ext  = bbox.get_extent()
            print(f"   Bounding Box:  {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} mm")

            # Mittlerer NN-Abstand → Validierung der Sampling-Annahme
            if n > 1:
                dists = pcd.compute_nearest_neighbor_distance()
                mean_d = float(np.mean(dists))
                p95_d  = float(np.percentile(dists, 95))
                print(f"   Ø NN-Abstand:  {mean_d:.3f} mm  (P95: {p95_d:.3f} mm)")

                if TARGET_SPACING is not None:
                    expected_nn = 0.5 * TARGET_SPACING
                    ratio = mean_d / expected_nn
                    print(f"\n🎯 Spacing-Validierung:")
                    print(f"   Ziel-Spacing:        {TARGET_SPACING:.3f} mm")
                    print(f"   Erwarteter Ø NN:     {expected_nn:.3f} mm  (≈ 0.5 × spacing)")
                    print(f"   Gemessener Ø NN:     {mean_d:.3f} mm")
                    print(f"   Verhältnis:          {ratio:.2f}x  "
                          f"({'ok' if 0.8 <= ratio <= 1.2 else 'Abweichung prüfen'})")

        except ImportError:
            print(f"\n💡 Tipp: open3d installieren für detaillierte Stats:")
            print(f"   uv add open3d")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Fehler nach {elapsed:.2f} s:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    force_utf8_output()
    success = test_conversion()
    print("\n" + "=" * 70)
    exit(0 if success else 1)