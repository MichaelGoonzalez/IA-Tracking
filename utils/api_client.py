import requests
import threading
import datetime
import json

API_URL = "https://selesoluciona.com/xcargo/seguimiento/newconteo"

def send_count_data(terminal_id, package_type, detection_time=None):
    """
    Envía los datos de conteo a la API de forma asíncrona (en un hilo separado)
    para no bloquear el procesamiento de video.
    
    :param terminal_id: ID de la cámara (ObjectId de MongoDB como string).
    :param package_type: Tipo de paquete detectado (String).
    :param detection_time: Fecha/hora de detección (datetime). Si es None, usa ahora.
    """
    if detection_time is None:
        detection_time = datetime.datetime.now()
    
    # Formatear datos según el schema requerido
    payload = {
        "detectionTime": detection_time.isoformat(),
        "tipoPaquete": package_type,
        "terminal": terminal_id,
        # "movimiento": ... (Opcional, no tenemos este dato por ahora)
    }

    # Iniciar hilo para el envío
    thread = threading.Thread(target=_send_request, args=(payload,))
    thread.daemon = True
    thread.start()

def _send_request(payload):
    try:
        # Log del envío
        print(f"\n[API OUT] Enviando datos a {API_URL}...")
        print(f"[API OUT] Payload: {json.dumps(payload, indent=2)}")
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers, timeout=5)
        
        # Log de la respuesta
        if response.status_code == 200 or response.status_code == 201:
            print(f"[API IN] ✅ Éxito ({response.status_code}): {response.text}")
        else:
            print(f"[API IN] ❌ Error ({response.status_code}): {response.text}")
            
    except Exception as e:
        print(f"[API ERROR] 💥 Excepción al enviar datos: {e}")
