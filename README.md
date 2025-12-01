# Sistema de Detección y Tracking Multi-Cámara con YOLOv8

Este proyecto implementa un sistema profesional de visión por computadora para la detección y seguimiento (tracking) de paquetes en entornos logísticos. Utiliza **YOLOv8** para detección robusta y algoritmos como **ByteTrack** para mantener la identidad de objetos a través de múltiples cámaras RTSP simultáneamente.

## 👁️ Visión General de la Arquitectura

El sistema está diseñado para operar en un ciclo de alto rendimiento y mejora continua:

1.  **Ingesta de Video Asíncrona**: Cada cámara RTSP es gestionada por un hilo independiente (`threading`) que mantiene el buffer de video limpio, garantizando latencia mínima cercana al tiempo real.
2.  **Motor de Inferencia IA**: Los frames de todas las cámaras se sincronizan y procesan en lote (batch processing) utilizando la potencia de la GPU (CUDA). Esto permite escalar el número de cámaras sin saturar el procesador.
3.  **Tracking Inteligente**: Se emplea el algoritmo ByteTrack para asociar detecciones entre fotogramas consecutivos, asignando IDs únicos a cada paquete y evitando duplicados o pérdidas momentáneas.
4.  **Ciclo de Aprendizaje Activo (Active Learning)**: El sistema incluye herramientas para extraer datos nuevos automáticamente, permitiendo re-entrenar el modelo de forma incremental para adaptarse a nuevos tipos de paquetes o cambios de iluminación sin olvidar lo aprendido previamente.

## 🚀 Características Principales

-   **Soporte Multi-Cámara**: Conexión simultánea a múltiples streams RTSP definidos en configuración.
-   **Visualización Grid**: Panel de monitoreo unificado que muestra todas las cámaras en tiempo real.
-   **Procesamiento GPU Optimizado**: Inferencia en lote (batch) para maximizar el uso de hardware NVIDIA.
-   **Entrenamiento Incremental**: Capacidad de pausar, extraer nuevos datos y continuar entrenando el modelo sin perder conocimiento previo.
-   **Conteo Automático**: Sistema de conteo de objetos (paquetes) mediante cruce de líneas virtuales configurables (Horizontal/Vertical).
-   **Arquitectura Robusta**: Lectura de video asíncrona (threading) para minimizar latencia.

## 📋 Requisitos Previos

-   Python 3.10 o 3.11.
-   Tarjeta gráfica NVIDIA (Altamente recomendada).
-   Drivers CUDA instalados.

## 🛠️ Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/IA-Tracking.git
    cd IA-Tracking
    ```

2.  **Configurar Entorno Virtual:**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/Mac
    ```

3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Asegúrate de tener PyTorch con soporte CUDA instalado para rendimiento real).*

## ⚙️ Configuración

### 1. Definir Cámaras (.env)
Crea un archivo `.env` en la raíz del proyecto (basado en el ejemplo) y define tus cámaras separadas por comas:

```env
RTSP_CAMERAS="rtsp://admin:pass@ip:port/stream1,rtsp://admin:pass@ip:port/stream2"
```

### 2. Ajustes Generales y Conteo (config.yaml)
Edita `config.yaml` para:
-   Ajustar hiperparámetros de IA (confianza, modelo).
-   **Configurar Líneas de Conteo**: Define las coordenadas `[x1, y1, x2, y2]` para dibujar líneas virtuales en cada cámara y contar los paquetes que las cruzan.
    *(Ver comentarios dentro del archivo para ejemplos de líneas horizontales/verticales).*

## 🎮 Ejecución

El proyecto cuenta con un punto de entrada único para facilitar su uso:

```bash
venv\Scripts\python main.py
```

Esto iniciará el sistema, cargará el modelo entrenado, conectará todas las cámaras del `.env` y abrirá la ventana de monitoreo.

## 🧠 Entrenamiento y Mejora del Modelo

El sistema soporta un flujo de trabajo de mejora continua (Active Learning):

1.  **Captura de Datos**: Extrae frames automáticamente de tus cámaras RTSP para crear un dataset:
    ```bash
    venv\Scripts\python scripts/extract_frames.py
    ```
    *(Por defecto extrae 60 imágenes de cada cámara definida en .env)*

2.  **Etiquetado**: Usa herramientas como **LabelImg** para dibujar cajas en las imágenes guardadas en `data/raw_images`.

3.  **Preparación**: Organiza los nuevos datos junto con los existentes:
    ```bash
    venv\Scripts\python scripts/split_dataset.py --images data/raw_images --labels data/raw_labels
    ```

4.  **Re-Entrenamiento**:
    ```bash
    venv\Scripts\python scripts/train.py
    ```
    *El script detectará automáticamente el modelo anterior (`best.pt`) y continuará el entrenamiento desde ahí para refinar la precisión.*

## 🗂️ Estructura Clave

-   `main.py`: Punto de entrada principal.
-   `scripts/multi_cam_track.py`: Núcleo del tracking multi-cámara.
-   `scripts/train.py`: Lógica de entrenamiento incremental.
-   `data/`: Almacenamiento de datasets (imágenes y etiquetas).
-   `models/`: Pesos del modelo entrenado.
