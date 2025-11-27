import cv2
import threading
import os
import time
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
import sys

# Añadir el directorio raíz al path para poder importar utils si fuera necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Cargar variables de entorno desde .env (forzando ruta raíz)
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
loaded = load_dotenv(dotenv_path)
if not loaded:
    print(f"[WARN] No se pudo cargar el archivo .env en: {dotenv_path}")

class RTSPStream:
    """
    Clase para leer streams RTSP en un hilo separado.
    Esto evita que el procesamiento de frames bloquee la lectura y cause latencia/lag.
    """
    def __init__(self, url, cam_id):
        self.url = url
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(self.url)
        self.frame = None
        self.stopped = False
        self.connected = self.cap.isOpened()
        if not self.connected:
            print(f"[ERROR] No se pudo conectar a la cámara {cam_id}")
        else:
            print(f"[INFO] Conectado a cámara {cam_id}")
        
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True # El hilo muere si el programa principal muere

    def start(self):
        if self.connected:
            self.t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.connected:
                # Intentar reconexión básica
                time.sleep(5)
                try:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.url)
                    self.connected = self.cap.isOpened()
                    if self.connected:
                        print(f"[INFO] Reconectado a cámara {self.cam_id}")
                except Exception:
                    pass
                continue

            grabbed, frame = self.cap.read()
            if not grabbed:
                self.connected = False
                print(f"[WARN] Señal perdida de cámara {self.cam_id}")
                continue
            
            # Solo guardamos el último frame, descartando los anteriores para mantener tiempo real
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        if self.t.is_alive():
            self.t.join()
        self.cap.release()

def main():
    # 1. Cargar Configuración y Modelo
    model_path = "models/paquetes_tracking/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"[WARN] No se encontró modelo entrenado en {model_path}, usando yolov8n.pt base")
        model_path = "yolov8n.pt"
    
    print(f"[INFO] Cargando modelo: {model_path}")
    model = YOLO(model_path)

    # 2. Inicializar Cámaras
    streams = []
    
    # Intentar leer formato de lista separada por comas (RTSP_CAMERAS)
    cameras_env = os.getenv("RTSP_CAMERAS")
    print(f"[DEBUG] Valor crudo de RTSP_CAMERAS: {cameras_env}")
    
    if cameras_env:
        # Dividir por comas y limpiar
        urls = [u.strip().strip('"').strip("'") for u in cameras_env.split(',') if u.strip()]
        for i, url in enumerate(urls, 1):
            print(f"[INFO] Inicializando cámara {i}...")
            stream = RTSPStream(url, i)
            streams.append(stream)
    else:
        # Fallback a formato antiguo RTSP_CAM_1, RTSP_CAM_2...
        i = 1
        while True:
            url = os.getenv(f"RTSP_CAM_{i}")
            if not url:
                break
            url = url.strip('"').strip("'")
            if url:
                print(f"[INFO] Inicializando cámara {i}...")
                stream = RTSPStream(url, i)
                streams.append(stream)
            i += 1

    if not streams:
        print("[ERROR] No se encontraron cámaras configuradas en el archivo .env (Variable RTSP_CAMERAS o RTSP_CAM_X)")
        return

    # Iniciar hilos de lectura
    for stream in streams:
        stream.start()

    print("[INFO] Iniciando bucle principal de procesamiento...")
    
    # Configuración de visualización (Grid)
    # Calculamos el tamaño del grid (ej. 7 cams -> 3x3 grid)
    num_cams = len(streams)
    cols = 3 # Configurable: número de columnas
    rows = (num_cams + cols - 1) // cols
    
    # Tamaño objetivo para redimensionar cada cámara en el grid
    target_w, target_h = 640, 360 

    try:
        while True:
            frames_to_process = []
            active_streams_indices = []

            # Recolectar frames actuales
            for idx, stream in enumerate(streams):
                frame = stream.read()
                if frame is not None:
                    frames_to_process.append(frame)
                    active_streams_indices.append(idx)
                else:
                    # Si no hay frame listo, no procesamos nada de esta cámara en este ciclo
                    pass

            # Si no hay ningún frame activo, esperar un poco
            if not frames_to_process:
                time.sleep(0.01)
                # Mostrar pantalla de carga si no hay nada aún
                blank_screen = np.zeros((600, 800, 3), dtype=np.uint8)
                cv2.putText(blank_screen, "Esperando conexiones...", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Sistema Multi-Camara IA Tracking", blank_screen)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # INFERENCIA BATCH (Todo en un solo paso de GPU para eficiencia)
            # Procesamos todos los frames activos juntos
            results = model.track(
                source=frames_to_process, 
                persist=True, 
                conf=0.25, 
                iou=0.45, 
                tracker="bytetrack.yaml", 
                verbose=False
            )

            # --- Construcción del Grid de Visualización ---
            
            # Crear lista completa de frames para el grid (incluyendo los inactivos/negros)
            final_display_frames = [None] * num_cams
            
            # 1. Rellenar inactivos con negro
            for i in range(num_cams):
                blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                status_text = "NO SIGNAL / CONNECTING..."
                color = (0, 0, 255) # Rojo
                
                # Si la cámara está conectada pero no dio frame en este instante preciso
                if streams[i].connected:
                    status_text = "NO FRAME"
                    color = (0, 255, 255) # Amarillo

                cv2.putText(blank, f"CAM {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank, status_text, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                final_display_frames[i] = blank

            # 2. Rellenar activos con los resultados procesados
            # results es una lista correspondiente a frames_to_process
            for i, r in enumerate(results):
                # r.plot() devuelve el frame BGR con anotaciones dibujadas
                annotated_frame = r.plot()
                
                # Redimensionar para el grid
                annotated_frame = cv2.resize(annotated_frame, (target_w, target_h))
                
                # Recuperar índice original de la cámara
                original_cam_idx = active_streams_indices[i]
                
                # Añadir etiqueta de cámara sobre el video
                cv2.putText(annotated_frame, f"CAM {streams[original_cam_idx].cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                final_display_frames[original_cam_idx] = annotated_frame

            # 3. Ensamblar Grid
            grid_rows = []
            for r in range(rows):
                # Obtener frames de esta fila
                start_idx = r * cols
                end_idx = min((r + 1) * cols, num_cams)
                row_frames = final_display_frames[start_idx : end_idx]
                
                # Si la fila está incompleta (ej. última fila con 1 cámara de 3), rellenar con negro
                while len(row_frames) < cols:
                    row_frames.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
                
                # Unir horizontalmente
                grid_rows.append(np.hstack(row_frames))
            
            # Unir verticalmente las filas
            if len(grid_rows) > 0:
                final_grid = np.vstack(grid_rows)
            else:
                final_grid = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # 4. Mostrar Grid
            # Reducir un poco si es muy grande para la pantalla
            if final_grid.shape[0] > 1000:
                scale_factor = 1000 / final_grid.shape[0]
                final_grid = cv2.resize(final_grid, None, fx=scale_factor, fy=scale_factor)

            cv2.imshow("Sistema Multi-Camara IA Tracking", final_grid)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupción de teclado recibida.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
    finally:
        # Limpieza
        print("[INFO] Deteniendo streams...")
        for stream in streams:
            stream.stop()
        cv2.destroyAllWindows()
        print("[INFO] Finalizado.")

if __name__ == "__main__":
    main()
