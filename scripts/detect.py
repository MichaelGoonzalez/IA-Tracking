import sys
import os
import argparse
import cv2

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from utils.utils import load_config, get_device
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_detection(source, weights=None, save_result=True):
    config = load_config("config.yaml")
    device = get_device(config.get("device"))
    
    # Determinar qué pesos usar: argumento > config > default (best.pt del entrenamiento)
    if weights is None:
        # Intentar buscar el mejor modelo entrenado
        trained_weights = os.path.join(
            config.get("output_dir", "models/"), 
            config.get("project_name", "paquetes_tracking"), 
            "weights/best.pt"
        )
        if os.path.exists(trained_weights):
            weights = trained_weights
            logger.info(f"Usando pesos entrenados: {weights}")
        else:
            weights = "yolov8n.pt" # Fallback a modelo base
            logger.warning(f"No se encontraron pesos entrenados en {trained_weights}. Usando modelo base {weights}")
    
    model = YOLO(weights)
    
    logger.info(f"Iniciando detección y tracking en: {source}")
    
    # Ejecutar inferencia con TRACKING (model.track en lugar de model.predict)
    # stream=True es eficiente para videos/streams largos
    # persist=True es necesario para mantener IDs entre frames en videos
    tracker_type = config.get("tracker_type", "bytetrack.yaml")
    
    results = model.track(
        source=source,
        conf=config.get("conf_threshold", 0.25),
        iou=config.get("iou_threshold", 0.45),
        device=device,
        tracker=tracker_type,
        persist=True,
        show=True, # Mostrar video en tiempo real
        save=save_result, # Guarda imágenes/video anotado
        project="runs/detect_track", # Directorio por defecto para guardar resultados
        name="paquetes_output",
        exist_ok=True
    )
    
    logger.info("Proceso finalizado. Resultados guardados en runs/detect_track/paquetes_output")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Detección y Tracking de Paquetes")
    # Se establece prueba.mp4 como default para facilitar ejecución
    parser.add_argument("--source", type=str, default="prueba.mp4", help="Ruta de imagen, video o '0' para webcam")
    parser.add_argument("--weights", type=str, default=None, help="Ruta al archivo .pt del modelo")
    
    args = parser.parse_args()
    
    # Manejar input numérico para webcam
    source = args.source
    if source.isdigit():
        source = int(source)
        
    run_detection(source, args.weights)
