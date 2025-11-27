import sys
import os

# Añadir el directorio scripts al path para poder importar módulos desde allí
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Importar el script principal de tracking multi-cámara
from scripts.multi_cam_track import main

if __name__ == "__main__":
    print("[INFO] Iniciando Sistema IA Tracking...")
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Sistema detenido por el usuario.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error crítico: {e}")
        input("Presione Enter para salir...")
