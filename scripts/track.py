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

def run_tracking(source, weights=None):
    config = load_config("config.yaml")
    device = get_device(config.get("device"))
    tracker_file = config.get("tracker_type", "bytetrack.yaml")
    
    if weights is None:
        trained_weights = os.path.join(
            config.get("output_dir", "models/"), 
            config.get("project_name", "paquetes_tracking"), 
            "weights/best.pt"
        )
        if os.path.exists(trained_weights):
            weights = trained_weights
            logger.info(f"Usando pesos entrenados: {weights}")
        else:
            weights = "yolov8n.pt"
            logger.warning(f"Usando modelo base {weights}")

    model = YOLO(weights)
    
    logger.info(f"Iniciando tracking en: {source} con tracker: {tracker_file}")

    # El método track() de YOLOv8 maneja la detección y el seguimiento.
    # Soporta trackers como BoT-SORT y ByteTrack.
    # persist=True es crucial para video/streams para mantener IDs entre frames.
    
    results = model.track(
        source=source,
        conf=config.get("conf_threshold", 0.25),
        iou=config.get("iou_threshold", 0.45),
        device=device,
        tracker=tracker_file, # 'bytetrack.yaml' o 'botsort.yaml'
        persist=True,
        save=True, # Guardar video con tracking visualizado
        project="runs/track",
        name="paquetes_track",
        exist_ok=True
    )
    
    logger.info("Tracking finalizado. Resultados guardados en runs/track/paquetes_track")
    
    # Procesar resultados (opcional)
    # Por ejemplo, contar paquetes únicos
    # unique_ids = set()
    # for result in results:
    #     if result.boxes.id is not None:
    #         ids = result.boxes.id.cpu().numpy().astype(int)
    #         for i in ids:
    #             unique_ids.add(i)
    # logger.info(f"Total paquetes únicos detectados: {len(unique_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Tracking de Paquetes")
    parser.add_argument("--source", type=str, required=True, help="Ruta de video o '0' para webcam")
    parser.add_argument("--weights", type=str, default=None, help="Ruta al archivo .pt del modelo")
    
    args = parser.parse_args()
    
    source = args.source
    if source.isdigit():
        source = int(source)
    
    run_tracking(source, args.weights)
