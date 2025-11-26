# Guía de Etiquetado para el Video de la Banda Transportadora

Para que el modelo aprenda a reconocer los paquetes en tu video, necesitamos seguir estos pasos:

## 1. Extraer Imágenes del Video
Primero, vamos a convertir tu video en imágenes individuales (frames). No necesitamos todas, solo una muestra periódica.

1. Abre una terminal.
2. Ejecuta el script de extracción:
   ```bash
   python scripts/extract_frames.py --video prueba.mp4 --output data/raw_images --interval 10
   ```
   *   `--interval 10`: Guardará 1 frame de cada 10. Si el video es muy corto, baja este número (ej. 5). Si es muy largo, súbelo (ej. 30).
   *   Las imágenes se guardarán en `data/raw_images`.

## 2. Etiquetar las Imágenes (Label Labeling)
Necesitamos dibujar cuadros alrededor de los paquetes en cada imagen para decirle a la IA "esto es un paquete".

**Herramienta recomendada: LabelImg o CVAT**
Para empezar rápido y localmente, usa **LabelImg**.

### Instalación de LabelImg
```bash
pip install labelImg
```
O descargalo desde [su repositorio](https://github.com/heartexlabs/labelImg).

### Pasos para Etiquetar
1.  Ejecuta `labelImg` en tu terminal.
2.  **Open Dir**: Selecciona la carpeta `data/raw_images`.
3.  **Change Save Dir**: Crea una carpeta `data/raw_labels` y selecciónala.
4.  **Importante**: En la barra lateral, asegúrate de que el formato sea **YOLO** (no PascalVOC).
5.  Empieza a etiquetar:
    *   Presiona `W` para crear un cuadro (`rectbox`).
    *   Dibuja el cuadro alrededor del paquete.
    *   Escribe `paquete` como nombre de la clase (o `0`).
    *   Presiona `D` para pasar a la siguiente imagen.
    *   Repite hasta terminar todas las imágenes.

## 3. Organizar el Dataset
YOLO necesita una estructura específica para entrenar. Vamos a dividir las imágenes y etiquetas en conjuntos de entrenamiento (`train`) y validación (`val`).

Estructura final esperada:
```
data/
├── images/
│   ├── train/  (80% de las imágenes)
│   └── val/    (20% de las imágenes)
└── labels/
    ├── train/  (80% de los .txt correspondientes)
    └── val/    (20% de los .txt correspondientes)
```

**Script Automático (Opcional pero Recomendado)**
Puedes mover los archivos manualmente, o crear un pequeño script de python para dividir `raw_images` y `raw_labels` en train/val aleatoriamente.

## 4. Actualizar Configuración
Una vez organizadas las carpetas, verifica `data/dataset.yaml`:

```yaml
path: ../data
train: images/train
val: images/val

names:
  0: paquete
```

## 5. Entrenar
Ahora sí, ejecuta el entrenamiento:
```bash
python scripts/train.py
