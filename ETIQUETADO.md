# Guía de Etiquetado y Mejora Continua del Modelo

Para mejorar la precisión del modelo en tu entorno específico, sigue estos pasos para capturar nuevos datos, etiquetarlos y re-entrenar.

## 1. Capturar Imágenes (Data Mining)

En lugar de usar videos sueltos, ahora puedes capturar imágenes directamente de tus cámaras RTSP configuradas.

**Opción Automática (Recomendada):**
Extrae una muestra de imágenes de TODAS las cámaras configuradas en tu `.env`.
```bash
venv\Scripts\python scripts/extract_frames.py
```
*   Esto guardará 60 imágenes por cámara en `data/raw_images`.
*   Las imágenes se toman cada 30 cuadros para asegurar variedad.

**Opción Manual:**
Si prefieres usar un video grabado:
```bash
venv\Scripts\python scripts/extract_frames.py --video tu_video.mp4
```

## 2. Etiquetar las Imágenes

Usa **LabelImg** para dibujar cajas alrededor de los paquetes en las nuevas imágenes.

1.  Abre LabelImg:
    ```bash
    labelimg
    ```
2.  **Open Dir**: Selecciona `data/raw_images`.
3.  **Change Save Dir**: Selecciona `data/raw_labels`.
4.  **Formato**: Asegúrate de que esté en **YOLO** (barra lateral).
5.  Etiqueta los paquetes en cada imagen.

## 3. Integrar y Entrenar

Una vez etiquetadas, integra los nuevos datos al dataset principal y entrena.

1.  **Organizar Dataset**:
    Este comando toma tus nuevas imágenes/labels y las mezcla con el dataset histórico en `data/images`.
    ```bash
    venv\Scripts\python scripts/split_dataset.py --images data/raw_images --labels data/raw_labels
    ```

2.  **Re-Entrenar Modelo**:
    ```bash
    venv\Scripts\python scripts/train.py
    ```
    *   El sistema detectará que ya existe un modelo (`best.pt`) y lo usará como base.
    *   Esto permite que el modelo aprenda de los nuevos casos sin olvidar lo anterior.

## 4. Limpieza (Opcional)

Si quieres empezar una nueva sesión de etiquetado desde cero (sin ver las imágenes que acabas de procesar en la carpeta raw), ejecuta:
```bash
venv\Scripts\python scripts/clean_raw_data.py
```
*Esto solo borra la carpeta temporal `raw`, no tus datos de entrenamiento consolidados.*
