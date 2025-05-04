#!/usr/bin/env python3
import os
import json
import io
import threading
from collections import deque

import cv2
import torch
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string, jsonify

# ---- Configuration ----
YOLO_MODEL_PATH = 'weights/yolov5m.pt'
WAVE_MODEL_PATH = 'weights/wave_sequence_model_final.keras'
CONF_THRESHOLD  = 0.5
CLIP_LENGTH     = 16
ROI_SIZE        = 224

# ---- Shared state ----
state_lock = threading.Lock()
latest = {
    'odometry': {},
    'wave_prob': 0.0
}
# Buffer of recent person-ROIs for wave clip
roi_buffer = deque(maxlen=CLIP_LENGTH)

# ---- Load models ----
print("Loading YOLO detector and wave model…")
detector = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
detector.conf = CONF_THRESHOLD
wave_model = tf.keras.models.load_model(WAVE_MODEL_PATH)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print("Models loaded.")

# ---- Helper: detect largest person box ----
def detect_person_box(frame):
    # frame is BGR; convert to RGB for YOLO
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, size=640).xyxy[0].cpu().numpy()
    # class 0 is “person”
    persons = [d for d in dets if int(d[5]) == 0 and d[4] >= CONF_THRESHOLD]
    if not persons:
        return None
    # pick highest-confidence box
    x1, y1, x2, y2, _, _ = max(persons, key=lambda d: d[4])
    return map(int, (x1, y1, x2, y2))

# ---- Flask app ----
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Live Wave Monitor</title>
    <meta http-equiv="refresh" content="1">
    <style>body{font-family:Arial,sans-serif;padding:20px;} .item{margin:8px 0;}</style>
  </head>
  <body>
    <h1>Latest Data</h1>
    {% if odom %}
      <div class="item"><strong>Latitude:</strong> {{ odom.lat }}</div>
      <div class="item"><strong>Longitude:</strong> {{ odom.lon }}</div>
      <div class="item"><strong>Altitude:</strong> {{ odom.alt }}</div>
    {% else %}
      <p>No odometry yet.</p>
    {% endif %}
    <div class="item"><strong>Wave Probability:</strong> {{ wave_prob | round(2) }}</div>
  </body>
</html>
"""

@app.route('/')
def index():
    with state_lock:
        data = latest.copy()
    return render_template_string(INDEX_HTML,
                                  odom=data['odometry'],
                                  wave_prob=data['wave_prob'])

@app.route('/receive', methods=['POST'])
def receive():
    global roi_buffer
    img_file = request.files.get('image')
    if not img_file:
        return "Missing image", 400

    # decode JPEG → BGR frame
    img_bytes = img_file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # parse odometry JSON
    odom = json.loads(request.form.get('odometry', '{}'))

    # run person detection
    box = detect_person_box(frame)
    if box:
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size:
            # RGB resize + buffer
            roi = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))
            roi_buffer.append(roi)

    # if we have a full clip, run wave model
    wave_p = None
    if len(roi_buffer) == CLIP_LENGTH:
        clip = np.stack(roi_buffer).astype('float32')
        clip = preprocess_input(clip)
        wave_p = float(wave_model.predict(clip[None, ...])[0,0])
        roi_buffer.clear()

    # update shared state
    with state_lock:
        latest['odometry'] = odom
        if wave_p is not None:
            latest['wave_prob'] = wave_p

    return "OK", 200

if __name__ == '__main__':
    # silence TF debug logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # serve on all interfaces
    app.run(host='0.0.0.0', port=5000, threaded=True)