import cv2
import os
import argparse
import sys
import time
from dotenv import load_dotenv

# Añadir directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Cargar variables de entorno forzando ruta raíz
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_from_source(source, output_dir, interval=30, limit=60, prefix="video"):
    """
    Extrae frames de una fuente (video o RTSP).
    """
    # Validar si es archivo local o URL
    is_url = str(source).startswith(("rtsp://", "http://", "https://"))
    
    if not is_url and not os.path.exists(source):
        logger.error(f"La fuente no existe: {source}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir la fuente: {source}")
        return

    frame_count = 0
    saved_count = 0
    
    logger.info(f"[{prefix}] Iniciando extracción. Objetivo: {limit} frames (cada {interval})...")

    try:
        while saved_count < limit:
            ret, frame = cap.read()
            
            if not ret:
                if is_url:
                    logger.warning(f"[{prefix}] Stream interrumpido o finalizado. Reintentando en 1s...")
                    time.sleep(1)
                    # Opcional: reconectar
                    continue
                else:
                    break
            
            if frame_count % interval == 0:
                filename = f"{prefix}_frame_{saved_count:04d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                cv2.imwrite(filepath, frame)
                saved_count += 1
                logger.info(f"[{prefix}] Guardado {saved_count}/{limit}: {filename}")
            
            frame_count += 1
            
            # Pequeña pausa para no saturar si es un stream muy rápido o bucle vacío
            if is_url:
                time.sleep(0.01) 

    except KeyboardInterrupt:
        logger.info("Interrupción de usuario.")
    finally:
        cap.release()
        logger.info(f"[{prefix}] Finalizado. Total guardado: {saved_count}")

def extract_from_env(output_dir, interval, limit):
    """
    Extrae frames de todas las cámaras definidas en el .env
    """
    cameras_env = os.getenv("RTSP_CAMERAS")
    if not cameras_env:
        logger.error("No se encontró la variable RTSP_CAMERAS en el archivo .env")
        return

    urls = [u.strip().strip('"').strip("'") for u in cameras_env.split(',') if u.strip()]
    
    if not urls:
        logger.error("No se encontraron URLs válidas en RTSP_CAMERAS")
        return

    logger.info(f"Se encontraron {len(urls)} cámaras en el .env")

    for i, url in enumerate(urls, 1):
        prefix = f"cam{i}"
        logger.info(f"Procesando Cámara {i}...")
        extract_from_source(url, output_dir, interval, limit, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraer frames de video o RTSP para dataset")
    parser.add_argument("--video", type=str, help="Ruta archivo video o URL RTSP. Si se omite y se usa --env, lee del .env")
    parser.add_argument("--env", action="store_true", help="Leer cámaras desde el archivo .env")
    parser.add_argument("--output", type=str, default="data/raw_images", help="Carpeta de salida")
    parser.add_argument("--interval", type=int, default=30, help="Guardar cada N frames (default: 30)")
    parser.add_argument("--limit", type=int, default=60, help="Límite de frames a guardar por fuente (default: 60)")
    
    args = parser.parse_args()
    
    # Lógica inteligente: Si se especifica video, usarlo. Si no, intentar usar .env por defecto.
    if args.video:
        prefix = os.path.splitext(os.path.basename(args.video))[0]
        if "rtsp" in args.video:
            prefix = "rtsp_stream"
        extract_from_source(args.video, args.output, args.interval, args.limit, prefix)
    else:
        # Por defecto usar el entorno si no hay video específico
        print("[INFO] No se especificó video, intentando leer cámaras desde .env...")
        extract_from_env(args.output, args.interval, args.limit)
