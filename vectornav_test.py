#!/usr/bin/env python3
import os
import threading
import time
from collections import deque

import cv2
import torch
import tensorflow as tf
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, stream_with_context
from pymavlink import mavutil
import serial.tools.list_ports

# ---- Configuration ----
YOLO_MODEL_PATH      = 'weights/yolov5m.pt'
WAVE_MODEL_PATH      = 'weights/wave_sequence_model_file_split.h5'
CONF_THRESHOLD       = 0.5
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
CLIP_LENGTH          = 8
ROI_SIZE             = 224

PIXHAWK_PORT         = '/dev/cu.usbmodem01'  # update if needed
PIXHAWK_BAUD         = 57600

# ---- Shared telemetry dict ----
TELEMETRY = {
    'battery': 'N/A',
    'altitude': 'N/A',
    'speed': 'N/A',
    'pos_x': 'N/A',
    'pos_y': 'N/A',
    'pos_z': 'N/A',
    'wave_prob': 0.0
}
TELEMETRY_LOCK = threading.Lock()

# ---- Load models ----
print("Loading YOLO detector and wave model…")
detector   = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
detector.conf = CONF_THRESHOLD
wave_model = tf.keras.models.load_model(WAVE_MODEL_PATH)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print("Models loaded.")

# ---- Initialize webcam ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

roi_buffer = deque(maxlen=CLIP_LENGTH)
latest_prob = 0.0

# ---- Flask app & HTML template ----
app = Flask(__name__)
HTML = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Drone Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ margin:0; display:flex; height:100vh; font-family:Arial,sans-serif; background:#f0f2f5; }}
    .video-container {{ flex:3; }}
    .info-panel {{ flex:1; padding:20px; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.1); overflow:auto; }}
    .info-panel h2 {{ margin-top:0; }}
    .telemetry-item {{ margin:10px 0; }}
    #deliverBtn {{ display:block; margin:20px 0; padding:10px 20px; background:#28a745; border:none; color:#fff; cursor:pointer; font-size:1em; border-radius:5px; }}
    #statusMsg {{ margin-top:10px; color:#007bff; }}
    img {{ width:100%; height:auto; display:block; }}
    canvas {{ width:100%; height:200px; margin-top:20px; }}
  </style>
</head>
<body>
  <div class="video-container">
    <img src="{{{{ url_for('video_feed') }}}}" width="{FRAME_WIDTH}" height="{FRAME_HEIGHT}" />
  </div>
  <div class="info-panel">
    <h2>Telemetry</h2>
    <div class="telemetry-item"><strong>Battery:</strong> <span id="battery">--</span></div>
    <div class="telemetry-item"><strong>Altitude:</strong> <span id="altitude">--</span></div>
    <div class="telemetry-item"><strong>Speed:</strong> <span id="speed">--</span></div>
    <div class="telemetry-item"><strong>Position X:</strong> <span id="pos_x">--</span></div>
    <div class="telemetry-item"><strong>Position Y:</strong> <span id="pos_y">--</span></div>
    <div class="telemetry-item"><strong>Position Z:</strong> <span id="pos_z">--</span></div>
    <div class="telemetry-item"><strong>Wave Prob:</strong> <span id="waveProb">0.00</span></div>
    <button id="deliverBtn">Deliver Aid</button>
    <div id="statusMsg"></div>
    <canvas id="odometryChart"></canvas>
  </div>

  <script>
    // Initialize Chart.js odometry plot
    const ctx = document.getElementById('odometryChart').getContext('2d');
    const odomChart = new Chart(ctx, {{
      type: 'line',
      data: {{ datasets: [{{ label: 'Odometry Path', data: [], borderColor:'rgb(54,162,235)', fill:false }}] }},
      options: {{
        animation: {{ duration:0 }},
        scales: {{
          x: {{ type:'linear', position:'bottom', title:{{ display:true, text:'X (m)' }} }},
          y: {{ title:{{ display:true, text:'Y (m)' }} }}
        }},
        plugins: {{ legend:{{ display:false }} }}
      }}
    }});

    // Fetch telemetry & update UI
    async function fetchTelemetry() {{
      const res = await fetch('/telemetry');
      const data = await res.json();

      document.getElementById('battery').textContent  = data.battery;
      document.getElementById('altitude').textContent = data.altitude;
      document.getElementById('speed').textContent    = data.speed;
      document.getElementById('pos_x').textContent    = data.pos_x;
      document.getElementById('pos_y').textContent    = data.pos_y;
      document.getElementById('pos_z').textContent    = data.pos_z;
      document.getElementById('waveProb').textContent = data.wave_prob.toFixed(2);

      // Update odometry chart
      const x = parseFloat(data.pos_x);
      const y = parseFloat(data.pos_y);
      if (!isNaN(x) && !isNaN(y)) {{
        odomChart.data.datasets[0].data.push({{x,y}});
        if (odomChart.data.datasets[0].data.length > 500) {{
          odomChart.data.datasets[0].data.shift();
        }}
        odomChart.update('none');
      }}
    }}

    document.getElementById('deliverBtn').onclick = async () => {{
      document.getElementById('statusMsg').textContent = 'Initiating delivery...';
      const resp = await fetch('/deliver', {{ method:'POST' }});
      const json = await resp.json();
      document.getElementById('statusMsg').textContent = json.status;
    }};

    setInterval(fetchTelemetry, 1000);
    fetchTelemetry();
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/telemetry')
def telemetry():
    with TELEMETRY_LOCK:
        TELEMETRY['wave_prob'] = latest_prob
        return jsonify(TELEMETRY)

@app.route('/deliver', methods=['POST'])
def deliver():
    print("[Action] Delivery initiated")
    return jsonify({'status':'Delivery initiated'}), 200

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(gen_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- Pixhawk thread ----
def find_pixhawk_port():
    for port in serial.tools.list_ports.comports():
        if any(tag in port.description for tag in ('PX4','Pixhawk')) or 'usbmodem' in port.device:
            return port.device
    return PIXHAWK_PORT

def pixhawk_thread():
    master = mavutil.mavlink_connection(find_pixhawk_port(), baud=PIXHAWK_BAUD)
    master.wait_heartbeat()
    while True:
        msg = master.recv_match(type=['SYS_STATUS','GLOBAL_POSITION_INT','VFR_HUD','LOCAL_POSITION_NED'], blocking=True)
        if not msg: continue
        with TELEMETRY_LOCK:
            t = msg.get_type()
            if t == 'SYS_STATUS':
                pct = max(msg.battery_remaining,0)
                TELEMETRY['battery'] = f"{pct}%"
            elif t == 'GLOBAL_POSITION_INT':
                TELEMETRY['altitude'] = f"{msg.relative_alt/1000:.1f}m"
            elif t == 'VFR_HUD':
                TELEMETRY['speed'] = f"{msg.groundspeed:.1f}m/s"
            elif t == 'LOCAL_POSITION_NED':
                TELEMETRY['pos_x'] = f"{msg.x:.2f}"
                TELEMETRY['pos_y'] = f"{msg.y:.2f}"
                TELEMETRY['pos_z'] = f"{msg.z:.2f}"

# Start telemetry thread
threading.Thread(target=pixhawk_thread, daemon=True).start()

# ---- Video / wave detection ----
def detect_person_box(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, size=640).xyxy[0].cpu().numpy()
    persons = [d for d in dets if int(d[5])==0 and d[4]>=CONF_THRESHOLD]
    if not persons: return None
    x1,y1,x2,y2,_,_ = max(persons, key=lambda d:d[4])
    return map(int, (x1,y1,x2,y2))

def gen_frames():
    global latest_prob
    while True:
        ret, frame = cap.read()
        if not ret: continue

        box = detect_person_box(frame)
        if box:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                roi = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (ROI_SIZE,ROI_SIZE))
                roi_buffer.append(roi)
                if len(roi_buffer)==CLIP_LENGTH:
                    clip = preprocess_input(np.stack(roi_buffer).astype('float32'))
                    latest_prob = float(wave_model.predict(clip[None])[0,0])
                    roi_buffer.clear()
            label = 'Waving' if latest_prob>=0.5 else 'Person'
            cv2.putText(frame, f"{label}", (x1,max(y1-10,20)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.putText(frame, f"Wave p={latest_prob:.2f}", (10,FRAME_HEIGHT-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

if __name__=='__main__':
    # Disable TensorFlow INFO logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    app.run(host='0.0.0.0', port=3000, threaded=True)