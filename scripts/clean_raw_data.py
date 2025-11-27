import os
import shutil
import sys

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_directory(directory):
    if not os.path.exists(directory):
        logger.warning(f"Directorio no encontrado: {directory}")
        return

    logger.info(f"Limpiando directorio: {directory}")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"No se pudo borrar {file_path}. Razón: {e}")

def main():
    print("ADVERTENCIA: Este script borrará TODAS las imágenes y etiquetas en:")
    print("  - data/raw_images/")
    print("  - data/raw_labels/")
    print("Esto es útil para iniciar una nueva sesión de etiquetado sin mezclar con los archivos crudos anteriores.")
    print("Nota: Los datos ya procesados en data/images/ y data/labels/ NO se tocarán.")
    
    confirm = input("¿Estás seguro de continuar? (s/n): ")
    if confirm.lower() != 's':
        print("Operación cancelada.")
        return

    clean_directory("data/raw_images")
    clean_directory("data/raw_labels")
    
    logger.info("Limpieza completada. Ahora puedes extraer nuevos frames y etiquetarlos.")

if __name__ == "__main__":
    main()
