import sys
import os
# Añadir el directorio raíz al path para poder importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from utils.utils import load_config, get_device
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    # Cargar configuración
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.error(f"No se pudo cargar la configuración: {e}")
        return

    # Configurar dispositivo
    device = get_device(config.get("device"))
    logger.info(f"Usando dispositivo: {device}")

    # Verificar dataset
    data_yaml = config.get("data_yaml")
    if not os.path.exists(data_yaml):
        logger.error(f"Archivo de dataset no encontrado en: {data_yaml}")
        logger.warning("Por favor asegúrese de tener un archivo dataset.yaml válido en la carpeta data/")
        # No retornamos aquí para permitir que YOLO intente descargar datasets de prueba si es el caso,
        # o fallará más adelante con un error claro.
    
    # Inicializar modelo
    # Verificar si existe un modelo entrenado previamente para continuar el entrenamiento
    project_name = config.get("project_name", "yolo_project")
    output_dir = config.get("output_dir", "models/")
    trained_weights = os.path.join(output_dir, project_name, "weights/best.pt")
    
    if os.path.exists(trained_weights):
        model_name = trained_weights
        logger.info(f"Encontrado modelo previo. Continuando entrenamiento desde: {model_name}")
    else:
        model_name = config.get("model", "yolov8n.pt")
        logger.info(f"No se encontró modelo previo. Iniciando desde base: {model_name}")
        
    model = YOLO(model_name)

    # Configurar argumentos de entrenamiento
    # Se pueden pasar muchos argumentos en el método train()
    # Mapeamos augmentations del config a argumentos de YOLO
    aug_config = config.get("augmentations", {})
    
    project_name = config.get("project_name", "yolo_project")
    output_dir = config.get("output_dir", "models/")

    logger.info("Iniciando entrenamiento...")
    
    try:
        results = model.train(
            data=data_yaml,
            imgsz=config.get("imgsz", 640),
            epochs=config.get("epochs", 50),
            batch=config.get("batch_size", 16),
            device=device,
            project=output_dir,
            name=project_name,
            exist_ok=True, # Sobrescribir si existe la carpeta del experimento
            
            # Augmentations
            degrees=aug_config.get("degrees", 0.0),
            scale=aug_config.get("scale", 0.5),
            shear=aug_config.get("shear", 0.0),
            perspective=aug_config.get("perspective", 0.0),
            flipud=aug_config.get("flipud", 0.0),
            fliplr=aug_config.get("fliplr", 0.5),
            mosaic=aug_config.get("mosaic", 1.0),
            mixup=aug_config.get("mixup", 0.0),
            
            verbose=True
        )
        logger.info("Entrenamiento completado exitosamente.")
        
        # Guardar resultados o realizar acciones post-entrenamiento si es necesario
        # Ultralytics guarda automáticamente en project/name/weights/best.pt
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    train_model()
