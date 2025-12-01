import cv2
import numpy as np

class LineCounter:
    """
    Clase para contar objetos que cruzan una línea definida.
    """
    def __init__(self, start_point, end_point, class_names):
        """
        :param start_point: Tupla (x, y) de inicio de la línea.
        :param end_point: Tupla (x, y) de fin de la línea.
        :param class_names: Diccionario de nombres de clases {0: 'paquete', ...}
        """
        self.start_point = start_point
        self.end_point = end_point
        self.class_names = class_names
        
        # Almacenar la posición anterior de los objetos: {track_id: (x, y)}
        self.track_history = {}
        
        # Contadores por clase: {class_name: count}
        self.counts = {name: 0 for name in class_names.values()}
        self.total_count = 0
        
        # Conjunto de IDs ya contados para evitar duplicados inmediatos 
        # (aunque con lógica de cruce de línea es menos necesario, es bueno tenerlo)
        self.counted_ids = set()

    def update(self, detections):
        """
        Actualiza el estado del contador con nuevas detecciones.
        :param detections: Lista de objetos detectados [(x1, y1, x2, y2, track_id, class_id), ...]
        """
        for x1, y1, x2, y2, track_id, class_id in detections:
            track_id = int(track_id)
            class_id = int(class_id)
            
            # Calcular centroide actual
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            current_point = (cx, cy)
            
            # Si ya conocemos este objeto, verificamos si cruzó la línea
            if track_id in self.track_history:
                prev_point = self.track_history[track_id]
                
                if self._has_crossed_line(prev_point, current_point):
                    if track_id not in self.counted_ids:
                        class_name = self.class_names.get(class_id, "unknown")
                        self.counts[class_name] = self.counts.get(class_name, 0) + 1
                        self.total_count += 1
                        self.counted_ids.add(track_id)
            
            # Actualizar historia
            self.track_history[track_id] = current_point

    def _has_crossed_line(self, point_a, point_b):
        """
        Verifica si el segmento entre point_a y point_b intersecta con la línea de conteo.
        Utiliza el producto cruz para determinar la orientación relativa.
        """
        # Línea de conteo: P1 -> P2
        p1 = self.start_point
        p2 = self.end_point
        
        # Convertir a vectores
        line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        
        # Vectores desde P1 a los puntos del objeto
        vec_a = np.array([point_a[0] - p1[0], point_a[1] - p1[1]])
        vec_b = np.array([point_b[0] - p1[0], point_b[1] - p1[1]])
        
        # Calcular producto cruz (2D) - determina el lado de la línea
        # cross(v, w) = vx*wy - vy*wx
        cross_a = np.cross(line_vec, vec_a)
        cross_b = np.cross(line_vec, vec_b)
        
        # Si los signos son opuestos, cruzó la línea infinita
        if np.sign(cross_a) != np.sign(cross_b) and cross_a != 0 and cross_b != 0:
            # Ahora verificar si está dentro del segmento de la línea de conteo
            # Proyección o bounding box check.
            # Simplificación: bounding box check de los segmentos
            x_min_l, x_max_l = min(p1[0], p2[0]), max(p1[0], p2[0])
            y_min_l, y_max_l = min(p1[1], p2[1]), max(p1[1], p2[1])
            
            x_min_o, x_max_o = min(point_a[0], point_b[0]), max(point_a[0], point_b[0])
            y_min_o, y_max_o = min(point_a[1], point_b[1]), max(point_a[1], point_b[1])
            
            if (x_max_l >= x_min_o and x_max_o >= x_min_l and
                y_max_l >= y_min_o and y_max_o >= y_min_l):
                return True
                
        return False

    def draw(self, frame):
        """
        Dibuja la línea y el contador en el frame.
        """
        # Dibujar línea amarilla
        cv2.line(frame, self.start_point, self.end_point, (0, 255, 255), 2)
        
        # Dibujar conteo total
        # Posición del texto: esquina superior izquierda o cerca de la línea
        text = f"Total: {self.total_count}"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Opcional: dibujar desglose por clase
        y_offset = 90
        for cls, count in self.counts.items():
            # Color del texto (B, G, R): (0, 0, 0) es Negro
            cv2.putText(frame, f"{cls}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 25
            
        return frame
