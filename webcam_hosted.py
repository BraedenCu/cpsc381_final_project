#!/usr/bin/env python3
"""
Flask app for live wave detection via webcam.
Streams processed frames to browser and displays drone telemetry.
Now uses ultralytics.YOLO for offline YOLOv5 loading.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template_string, stream_with_context, jsonify
from collections import deque
from ultralytics import YOLO  # pip install ultralytics

# suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Configuration ----
YOLO_MODEL_PATH = 'weights/yolov5m.pt'
WAVE_MODEL_PATH = 'weights/wave_sequence_model_final.keras'
CONF_THRESHOLD  = 0.5
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
CLIP_LENGTH     = 16
ROI_SIZE        = 224  # model input size

# ---- Telemetry storage ----
TELEMETRY = {
    'battery': 'N/A',
    'altitude': 'N/A',
    'speed': 'N/A',
    'wave_prob': 0.0
}

# ---- Load models once ----
print("Loading models...")
# load yolov5 model from local .pt (no internet needed)
detector = YOLO(YOLO_MODEL_PATH)
# load your Keras wave detector
wave_model = tf.keras.models.load_model(WAVE_MODEL_PATH)
print("Models loaded.")

# ---- Initialize webcam once ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# buffer for a sliding window of full frames
frame_buffer = deque(maxlen=CLIP_LENGTH)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
latest_prob = 0.0

# ---- Flask app & HTML ----
app = Flask(__name__)
HTML = f'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Drone Monitor</title>
  <style>
    body {{ margin:0; display:flex; height:100vh; font-family:Arial, sans-serif; background:#f0f2f5; }}
    .video-container {{ flex:3; position:relative; }}
    .info-panel {{ flex:1; padding:20px; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
    .info-panel h2 {{ margin-top:0; }}
    .telemetry-item {{ margin:10px 0; }}
    img {{ width:100%; height:auto; display:block; }}
  </style>
</head>
<body>
  <div class="video-container">
    <img src="{{{{ url_for('video_feed') }}}}" />
  </div>
  <div class="info-panel">
    <h2>Telemetry</h2>
    <div class="telemetry-item"><strong>Battery:</strong> <span id="battery">--</span></div>
    <div class="telemetry-item"><strong>Altitude:</strong> <span id="altitude">--</span></div>
    <div class="telemetry-item"><strong>Speed:</strong> <span id="speed">--</span></div>
    <div class="telemetry-item"><strong>Wave Prob:</strong> <span id="waveProb">0.00</span></div>
  </div>
  <script>
    async function fetchTelemetry() {{
      try {{
        const resp = await fetch('/telemetry');
        const data = await resp.json();
        document.getElementById('battery').textContent = data.battery;
        document.getElementById('altitude').textContent = data.altitude;
        document.getElementById('speed').textContent = data.speed;
        document.getElementById('waveProb').textContent = data.wave_prob.toFixed(2);
      }} catch (e) {{ console.error(e); }}
    }}
    setInterval(fetchTelemetry, 1000);
    fetchTelemetry();
  </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/telemetry')
def telemetry():
    TELEMETRY['wave_prob'] = latest_prob
    return jsonify(TELEMETRY)

# ---- Detection helper via ultralytics YOLO ----
def detect_person_box(frame):
    # frame is BGR; YOLO wants RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector(rgb)[0]           # first (and only) batch item
    persons = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            persons.append((x1, y1, x2, y2, conf))
    if not persons:
        return None
    # pick highest-confidence person
    x1, y1, x2, y2, _ = max(persons, key=lambda x: x[4])
    return x1, y1, x2, y2

# ---- Frame generator ----
def gen_frames():
    global latest_prob
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # draw person detection
        box = detect_person_box(frame)
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # feed full frame into sliding buffer
        rgb_full    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_full = cv2.resize(rgb_full, (ROI_SIZE, ROI_SIZE))
        frame_buffer.append(resized_full)

        if len(frame_buffer) == CLIP_LENGTH:
            clip = np.stack(frame_buffer, axis=0).astype('float32')
            clip = preprocess_input(clip)
            latest_prob = float(wave_model.predict(clip[None, ...])[0,0])
            frame_buffer.popleft()

        # overlay label
        if latest_prob >= 0.3:
            label, color = 'Waving', (0,255,0)
        else:
            label, color = 'Person', (0,255,0)
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Wave prob: {latest_prob:.2f}", (10, FRAME_HEIGHT-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # yield JPEG frame
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        stream_with_context(gen_frames()),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000, threaded=True)