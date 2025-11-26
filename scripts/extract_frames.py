import cv2
import os
import argparse
import sys

# Añadir directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_frames(video_path, output_dir, interval=30):
    """
    Extrae frames de un video y los guarda como imágenes.
    
    :param video_path: Ruta al archivo de video.
    :param output_dir: Directorio donde guardar las imágenes.
    :param interval: Guardar un frame cada 'interval' frames (para no tener demasiadas imágenes iguales).
    """
    
    if not os.path.exists(video_path):
        logger.error(f"El video no existe: {video_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Directorio creado: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir el video: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    logger.info(f"Iniciando extracción de frames del video: {video_path}")
    logger.info(f"Guardando 1 de cada {interval} frames...")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % interval == 0:
            # Nombre del archivo: nombrevideo_frame_0001.jpg
            filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, frame)
            saved_count += 1
            if saved_count % 10 == 0:
                logger.info(f"Guardadas {saved_count} imágenes...")
        
        frame_count += 1

    cap.release()
    logger.info(f"Extracción completada. Total imágenes guardadas: {saved_count} en {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraer frames de un video para dataset")
    parser.add_argument("--video", type=str, required=True, help="Ruta al archivo de video (ej: prueba.mp4)")
    parser.add_argument("--output", type=str, default="data/raw_images", help="Carpeta de salida para las imágenes")
    parser.add_argument("--interval", type=int, default=30, help="Guardar cada N frames (default: 30)")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.interval)
