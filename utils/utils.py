import yaml
import torch
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """
    Carga la configuración desde un archivo YAML.
    """
    if not os.path.exists(config_path):
        logger.error(f"Archivo de configuración no encontrado: {config_path}")
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            logger.info("Configuración cargada exitosamente.")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error al leer el archivo YAML: {e}")
            raise

def get_device(device_config=None):
    """
    Determina el dispositivo a utilizar (GPU o CPU).
    Verifica si torch.cuda.is_available() devuelve True.
    """
    if torch.cuda.is_available():
        if device_config == "cpu":
            logger.warning("GPU disponible pero se solicitó CPU en la configuración.")
            return "cpu"
        
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detectada y disponible: {gpu_name}")
        return "0" # Ultralytics usa "0", "1", etc. para indexar GPUs
    else:
        logger.info("GPU no detectada. Usando CPU.")
        return "cpu"

def check_paths(paths):
    """
    Verifica que las rutas existan.
    :param paths: Lista de rutas o diccionario de rutas a verificar.
    """
    if isinstance(paths, dict):
        path_list = paths.values()
    else:
        path_list = paths
        
    for p in path_list:
        if p and not os.path.exists(p) and not str(p).endswith('.pt'): # Excluir pesos pre-entrenados si se van a descargar
             # Nota: Para directorios de salida, quizás no sea necesario que existan, pero para inputs sí.
             # Aquí solo logueamos advertencias para no bloquear flujos donde se crean directorios al vuelo.
             logger.warning(f"Ruta no encontrada (puede que se cree durante la ejecución): {p}")
