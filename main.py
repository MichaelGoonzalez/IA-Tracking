import sys
import os
import argparse

# Añadir el directorio scripts al path para poder importar módulos desde allí
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Importar el script principal de tracking multi-cámara
from scripts.multi_cam_track import main

def parse_args():
    parser = argparse.ArgumentParser(description="Sistema IA Tracking - Detección y Conteo de Paquetes")
    # Aceptar cualquier argumento posicional o flags para el video
    parser.add_argument("video_path", nargs="?", help="Ruta al archivo de video para modo prueba (opcional)")
    parser.add_argument("--source", type=str, help="Ruta al archivo de video para modo prueba (alternativo)")
    parser.add_argument("--prueba.mp4", dest="prueba_mp4_flag", action="store_true", help="Flag para usar prueba.mp4 rápidamente")
    parser.add_argument("--no-gui", action="store_true", help="Ejecutar sin interfaz gráfica (modo servidor)")
    
    # Truco para soportar el formato no estándar --prueba.mp4 como si fuera un flag
    # Si detectamos un argumento que empieza por -- y termina en .mp4/.avi/etc, lo tratamos como source
    args, unknown = parser.parse_known_args()
    
    video_source = args.video_path or args.source
    
    # Manejar flags raros o directos como --prueba.mp4
    if args.prueba_mp4_flag:
        video_source = "prueba.mp4"
    
    # Revisar unknown args por si el usuario pasa --prueba.mp4 sin definirlo explícitamente en argparse
    for arg in unknown:
        if arg.startswith("--") and (arg.endswith(".mp4") or arg.endswith(".avi")):
            video_source = arg.lstrip("-") # quitamos los guiones
    
    return video_source, args.no_gui

if __name__ == "__main__":
    print("[INFO] Iniciando Sistema IA Tracking...")
    try:
        video_source, headless = parse_args()
        main(video_source=video_source, headless=headless)
    except KeyboardInterrupt:
        print("\n[INFO] Sistema detenido por el usuario.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error crítico: {e}")
        input("Presione Enter para salir...")
