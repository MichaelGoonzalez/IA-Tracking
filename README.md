# Sistema de Detección y Tracking de Paquetes con YOLOv8

Este proyecto implementa un sistema de visión por computadora avanzado para la detección y seguimiento (tracking) automático de paquetes en bandas transportadoras o entornos logísticos. Utiliza **YOLOv8** para una detección robusta y algoritmos de tracking como **ByteTrack** para mantener la identidad de los objetos a través del tiempo.

## 🚀 Características

-   **Detección en Tiempo Real**: Identifica paquetes con alta precisión incluso en movimiento.
-   **Tracking Continuo**: Asigna IDs únicos a cada paquete para conteo y seguimiento.
-   **Entrenamiento Personalizado**: Scripts listos para entrenar el modelo con tus propios datos.
-   **Soporte GPU**: Optimizado para usar aceleración NVIDIA CUDA si está disponible.
-   **Visualización en Vivo**: Muestra el video procesado con las cajas delimitadoras y trayectorias.

## 📋 Requisitos Previos

-   Python 3.8, 3.9, 3.10 o 3.11 (Recomendado: 3.10).
-   Tarjeta gráfica NVIDIA (Opcional pero altamente recomendada para entrenamiento rápido).
-   Drivers CUDA instalados (si se usa GPU).

## 🛠️ Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/IA-Tracking.git
    cd IA-Tracking
    ```

2.  **Crear un entorno virtual (Recomendado):**
    ```bash
    # En Windows
    python -m venv venv
    venv\Scripts\activate

    # En Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Si tienes una GPU NVIDIA, asegúrate de instalar la versión de PyTorch compatible con CUDA (ej. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`).*

## 🗂️ Estructura del Proyecto

```
IA-Tracking/
├── config.yaml          # Configuración central (rutas, hiperparámetros, tracking)
├── data/                # Dataset
│   ├── images/          # Imágenes de entrenamiento y validación
│   ├── labels/          # Etiquetas YOLO (.txt)
│   └── dataset.yaml     # Definición de clases y rutas para YOLO
├── models/              # Modelos entrenados (.pt)
├── scripts/             # Código fuente
│   ├── detect.py        # Script principal de inferencia y tracking
│   ├── train.py         # Script de entrenamiento
│   ├── extract_frames.py # Herramienta para extraer imágenes de videos
│   └── split_dataset.py  # Herramienta para organizar datasets
└── utils/               # Utilidades internas
```

## 🎮 Uso

### 1. Detección y Tracking (Inferencia)
Para probar el modelo con un video existente (por defecto busca `prueba.mp4`):

```bash
python scripts/detect.py
```

Para usar otro video o una webcam:
```bash
# Video específico
python scripts/detect.py --source ruta/a/tu/video.mp4

# Webcam en vivo
python scripts/detect.py --source 0
```
Se abrirá una ventana mostrando el análisis en tiempo real. Los resultados se guardarán en `runs/detect_track/`.

### 2. Entrenamiento de un Nuevo Modelo (Estable y Optimizado)
El sistema de entrenamiento es el núcleo más robusto del proyecto. Está diseñado para ser "Plug & Play": detecta tu hardware (CPU/GPU), carga la configuración y optimiza los hiperparámetros automáticamente.

#### Paso 1: Preparar tus Datos
1.  Coloca tus videos en la carpeta raíz o extrae imágenes directamente.
2.  Usa `scripts/extract_frames.py` para convertir videos en imágenes si es necesario.
3.  Etiqueta tus imágenes (usando LabelImg, Roboflow, etc.) y guárdalas en `data/raw_images` y `data/raw_labels`.

#### Paso 2: Organizar el Dataset
Ejecuta el script de organización. Este script valida tus datos, ignora imágenes sin etiquetas y crea la estructura de carpetas que YOLO necesita automáticamente:
```bash
python scripts/split_dataset.py --images data/raw_images --labels data/raw_labels
```

#### Paso 3: Iniciar Entrenamiento
Ejecuta el script maestro de entrenamiento:
```bash
python scripts/train.py
```
-   **Detección Automática de GPU**: Si tienes una tarjeta NVIDIA, el script la usará automáticamente para acelerar el proceso hasta 50x.
-   **Resultados**: Al finalizar, encontrarás tu modelo listo para usar en `models/paquetes_tracking/weights/best.pt`.
-   **Métricas**: Se generan gráficos de precisión y pérdida en la misma carpeta para evaluar el rendimiento.

## ⚙️ Configuración Avanzada
El archivo `config.yaml` permite ajustar:
-   **Hiperparámetros**: `epochs`, `batch_size`, `imgsz`.
-   **Aumentos de Datos**: `degrees` (rotación), `scale`, `flip`, etc., para hacer el modelo más robusto.
-   **Tracking**: Tipo de tracker (`bytetrack.yaml` o `botsort.yaml`) y umbrales de confianza.

## 📄 Licencia
Este proyecto es de código abierto y está disponible para uso educativo y comercial.
