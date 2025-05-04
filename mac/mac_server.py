#!/usr/bin/env python3
import os
import threading
import time
import cv2
import numpy as np
import requests
import torch
import tensorflow as tf
from collections import deque
from flask import Flask, render_template_string, Response, stream_with_context, jsonify

# ---- Configuration ----
JETSON_IP         = '172.20.10.2'
RAW_VIDEO_URL     = f'http://{JETSON_IP}:3000/video_feed'
TELEMETRY_SOURCE  = f'http://{JETSON_IP}:3000/telemetry'
YOLO_MODEL_PATH   = 'weights/yolov5m.pt'
WAVE_MODEL_PATH   = 'weights/wave_sequence_model_final.keras'
CONF_THRESHOLD    = 0.5
CLIP_LENGTH       = 16
ROI_SIZE          = 224
FETCH_INTERVAL    = 0.05   # seconds

TELEMETRY_KEYS = [
    'battery_pct','battery_voltage','battery_current',
    'throttle','speed','climb',
    'altitude_rel','altitude_abs','heading',
    'lat','lon','fix_type','satellites',
    'pos_x','pos_y','pos_z',
    'roll','pitch','yaw',
    'acc_x','acc_y','acc_z','hdop'
]

# ---- Shared state ----
state_lock        = threading.Lock()
latest_frame      = None
latest_telemetry  = {k: '––' for k in TELEMETRY_KEYS}
latest_wave_prob  = 0.0
roi_buffer        = deque(maxlen=CLIP_LENGTH)

# ---- Load models ----
print("Loading models on MacBook…")
detector = torch.hub.load('ultralytics/yolov5', 'custom',
                          path=YOLO_MODEL_PATH, force_reload=False)
detector.conf = CONF_THRESHOLD
wave_model = tf.keras.models.load_model(WAVE_MODEL_PATH)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print("Models loaded.")

def detect_person_box(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, size=640).xyxy[0].cpu().numpy()
    persons = [d for d in dets if int(d[5]) == 0 and d[4] >= CONF_THRESHOLD]
    if not persons:
        return None
    x1, y1, x2, y2, _, _ = max(persons, key=lambda d: d[4])
    return map(int, (x1, y1, x2, y2))

def frame_fetcher():
    global latest_frame
    while True:
        try:
            resp = requests.get(RAW_VIDEO_URL, stream=True, timeout=5)
            if not resp.ok:
                time.sleep(1)
                continue
            buf = b''
            for chunk in resp.iter_content(chunk_size=1024):
                buf += chunk
                start = buf.find(b'\xff\xd8')
                end   = buf.find(b'\xff\xd9')
                if start != -1 and end != -1 and end > start:
                    jpg = buf[start:end+2]
                    buf = buf[end+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with state_lock:
                            latest_frame = frame
        except Exception:
            time.sleep(1)

def telemetry_fetcher():
    global latest_telemetry
    while True:
        try:
            r = requests.get(TELEMETRY_SOURCE, timeout=1)
            if r.ok:
                data = r.json()
                with state_lock:
                    for k in TELEMETRY_KEYS:
                        if k in data:
                            latest_telemetry[k] = data[k]
        except Exception:
            pass
        time.sleep(FETCH_INTERVAL)

def processing_loop():
    global latest_wave_prob
    while True:
        with state_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            box = detect_person_box(frame)
            if box:
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    roi = cv2.resize(rgb, (ROI_SIZE, ROI_SIZE))
                    roi_buffer.append(roi)
            if len(roi_buffer) == CLIP_LENGTH:
                clip = np.stack(roi_buffer).astype('float32')
                clip = preprocess_input(clip)
                p = float(wave_model.predict(clip[None, ...])[0,0])
                with state_lock:
                    latest_wave_prob = p
                roi_buffer.clear()
        time.sleep(FETCH_INTERVAL)

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Drone Live Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap 5 CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background:#f8f9fa; }
    .card-video { height: 100%; }
    #video { border-radius: .25rem; }
  </style>
</head>
<body>
  <div class="container py-4">
    <h1 class="mb-4 text-center">🚁 Drone Live Monitor</h1>
    <div class="row gy-4">
      <div class="col-lg-8">
        <div class="card card-video shadow-sm">
          <div class="card-header bg-dark text-white">Live Video Feed</div>
          <div class="card-body p-0">
            <img id="video" src="{{ url_for('video_feed') }}" class="img-fluid w-100" alt="Video stream">
          </div>
        </div>
      </div>
      <div class="col-lg-4">
        <div class="card shadow-sm mb-4">
          <div class="card-header">Wave Probability</div>
          <div class="card-body">
            <div class="progress mb-2" style="height: 1.75rem;">
              <div id="waveBar" class="progress-bar" role="progressbar" style="width: 0%">0%</div>
            </div>
            <small class="text-muted">Updates every second</small>
          </div>
        </div>
        <div class="card shadow-sm">
          <div class="card-header">Odometry & Telemetry</div>
          <ul class="list-group list-group-flush" id="telemetryList">
            {% for key in keys %}
              <li class="list-group-item d-flex justify-content-between">
                <span class="text-capitalize">{{ key.replace('_',' ') }}</span>
                <span id="{{ key }}"> {{ telemetry[key] }} </span>
              </li>
            {% endfor %}
          </ul>
        </div>
      </div>
    </div>
  </div>

  <!-- Bootstrap 5 JS Bundle (includes Popper) -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    const TELEMETRY_KEYS = {{ keys|tojson }};
    async function updateTelemetry() {
      try {
        const res = await fetch('/telemetry');
        const data = await res.json();
        TELEMETRY_KEYS.forEach(k => {
          const el = document.getElementById(k);
          if (el && data[k] !== undefined) el.textContent = data[k];
        });
        // update wave bar
        const wp = data.wave_prob ?? 0;
        const pct = Math.round(wp * 100);
        const bar = document.getElementById('waveBar');
        bar.style.width = pct + '%';
        bar.textContent = pct + '%';
      } catch (e) {
        console.error(e);
      }
    }
    setInterval(updateTelemetry, 1000);
    window.addEventListener('load', updateTelemetry);
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    with state_lock:
        telem = latest_telemetry.copy()
        wp    = latest_wave_prob
    return render_template_string(
        INDEX_HTML,
        telemetry=telem,
        wave_prob=wp,
        keys=TELEMETRY_KEYS
    )

@app.route('/telemetry')
def telemetry():
    with state_lock:
        data = latest_telemetry.copy()
        data['wave_prob'] = latest_wave_prob
    return jsonify(data)

def gen_processed_frames():
    while True:
        with state_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(FETCH_INTERVAL)
            continue
        proc = frame.copy()
        box = detect_person_box(proc)
        if box:
            x1,y1,x2,y2 = box
            cv2.rectangle(proc, (x1,y1),(x2,y2),(0,255,0),2)
        ret, buf = cv2.imencode('.jpg', proc)
        if not ret:
            continue
        yield (
          b'--frame\r\n'
          b'Content-Type: image/jpeg\r\n\r\n' +
          buf.tobytes() +
          b'\r\n'
        )
        time.sleep(FETCH_INTERVAL)

@app.route('/video_feed')
def video_feed():
    return Response(
      stream_with_context(gen_processed_frames()),
      mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import warnings; warnings.filterwarnings('ignore', category=FutureWarning)
    threading.Thread(target=frame_fetcher,     daemon=True).start()
    threading.Thread(target=telemetry_fetcher, daemon=True).start()
    threading.Thread(target=processing_loop,    daemon=True).start()
    app.run(host='0.0.0.0', port=3113, threaded=True)