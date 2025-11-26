# -*- coding: utf-8 -*-
import os
import shutil
import random
import argparse
import logging
import sys

# Añadir raíz al path para configuración de logging si fuera necesario, 
# aunque aquí usamos logging básico directo.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(images_source, labels_source, dest_dir, split_ratio=0.8):
    """
    Divide datasets de imagenes y etiquetas en conjuntos de train y val.
    
    :param images_source: Carpeta con todas las imagenes.
    :param labels_source: Carpeta con todas las etiquetas (.txt).
    :param dest_dir: Carpeta destino (ej: data/).
    :param split_ratio: Porcentaje para entrenamiento (0.0 - 1.0).
    """
    
    # Crear estructura de carpetas
    train_images_dir = os.path.join(dest_dir, 'images', 'train')
    val_images_dir = os.path.join(dest_dir, 'images', 'val')
    train_labels_dir = os.path.join(dest_dir, 'labels', 'train')
    val_labels_dir = os.path.join(dest_dir, 'labels', 'val')

    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)

    # Listar imagenes
    images = [f for f in os.listdir(images_source) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not images:
        logger.error(f"No se encontraron imagenes en {images_source}")
        return

    # Emparejar con etiquetas
    pairs = []
    for img_file in images:
        name_no_ext = os.path.splitext(img_file)[0]
        label_file = name_no_ext + ".txt"
        label_path = os.path.join(labels_source, label_file)
        
        if os.path.exists(label_path):
            pairs.append((img_file, label_file))
        else:
            logger.warning(f"Imagen sin etiqueta encontrada: {img_file}. Se omitira.")

    if not pairs:
        logger.error("No se encontraron pares validos de imagen-etiqueta.")
        return

    # Aleatorizar
    random.shuffle(pairs)
    
    # Dividir
    split_index = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]
    
    logger.info(f"Total pares validos: {len(pairs)}")
    logger.info(f"Entrenamiento: {len(train_pairs)}")
    logger.info(f"Validacion: {len(val_pairs)}")

    # Copiar archivos
    def copy_files(file_pairs, img_dest, lbl_dest):
        for img, lbl in file_pairs:
            shutil.copy2(os.path.join(images_source, img), os.path.join(img_dest, img))
            shutil.copy2(os.path.join(labels_source, lbl), os.path.join(lbl_dest, lbl))

    logger.info("Copiando archivos de entrenamiento...")
    copy_files(train_pairs, train_images_dir, train_labels_dir)
    
    logger.info("Copiando archivos de validacion...")
    copy_files(val_pairs, val_images_dir, val_labels_dir)

    logger.info("Dataset dividido exitosamente!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dividir dataset en train y val para YOLO")
    parser.add_argument("--images", default="data/raw_images", help="Carpeta de imágenes fuente")
    parser.add_argument("--labels", default="data/raw_labels", help="Carpeta de etiquetas fuente")
    parser.add_argument("--dest", default="data", help="Directorio destino (raiz del dataset)")
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio de entrenamiento (0.0-1.0)")
    
    args = parser.parse_args()
    
    split_dataset(args.images, args.labels, args.dest, args.ratio)
