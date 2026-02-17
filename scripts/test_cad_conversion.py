#!/usr/bin/env python3
"""Test CAD API: STEP → PLY Konvertierung"""

from pathlib import Path
from client.core import CADConverterClient
import time

def test_conversion():
    # Client initialisieren
    client = CADConverterClient()
    
    # Pfade
    step_file = Path("external/cad-api-client/testdata/geometry_00000005.STEP")
    output_dir = Path("data/processed/test_geometry_00000005")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pointcloud.ply"
    
    print("=" * 70)
    print("CAD API Test: STEP → PLY Konvertierung")
    print("=" * 70)
    print(f"\n📁 Input:  {step_file}")
    print(f"📁 Output: {output_file}")
    
    if not step_file.exists():
        print(f"\n❌ STEP-Datei nicht gefunden: {step_file}")
        return False
    
    # Konvertierung mit Zeitmessung
    print(f"\n🔄 Starte Konvertierung...")
    start_time = time.time()
    
    try:
        # API Call
        path = client.convert_to_ply(str(step_file), str(output_file))
        elapsed = time.time() - start_time
        
        print(f"\n✅ Konvertierung erfolgreich!")
        print(f"   PLY gespeichert: {path.absolute()}")
        print(f"   Dauer: {elapsed:.2f} Sekunden")
        
        # File Info
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\n📊 Output File Info:")
        print(f"   Größe: {size_mb:.2f} MB")
        print(f"   Pfad: {path}")
        
        # Point Cloud Stats (mit open3d falls installiert)
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            print(f"\n📐 Point Cloud Stats:")
            print(f"   Punkte: {len(pcd.points):,}")
            print(f"   Hat Normalen: {pcd.has_normals()}")
            print(f"   Hat Farben: {pcd.has_colors()}")
            
            # Bounding Box
            bbox = pcd.get_axis_aligned_bounding_box()
            print(f"   Bounding Box: {bbox.get_extent()}")
            
        except ImportError:
            print(f"\n💡 Tipp: Installiere open3d für detaillierte Stats:")
            print(f"   uv add open3d")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Fehler nach {elapsed:.2f} Sekunden:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_conversion()
    print("\n" + "=" * 70)
    exit(0 if success else 1)