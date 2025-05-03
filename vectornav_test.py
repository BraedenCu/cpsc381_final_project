#!/usr/bin/env python3
import os
import threading
import time
from collections import deque

import cv2
import torch
import tensorflow as tf
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, stream_with_context, url_for
from pymavlink import mavutil
import serial.tools.list_ports

# ---- Configuration ----
YOLO_MODEL_PATH    = 'weights/yolov5m.pt'
WAVE_MODEL_PATH    = 'weights/wave_sequence_model_file_split.h5'
CONF_THRESHOLD     = 0.5
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
CLIP_LENGTH        = 8
ROI_SIZE           = 224

PIXHAWK_PORT       = '/dev/cu.usbmodem01'
PIXHAWK_BAUD       = 57600

# ---- Shared telemetry ----
TELEMETRY = {
    'battery_pct':     'N/A',
    'battery_voltage': 'N/A',
    'battery_current': 'N/A',
    'throttle':        'N/A',
    'speed':           'N/A',
    'climb':           'N/A',
    'altitude_rel':    'N/A',
    'altitude_abs':    'N/A',
    'heading':         'N/A',
    'lat':             'N/A',
    'lon':             'N/A',
    'fix_type':        'N/A',
    'satellites':      'N/A',
    'pos_x':           'N/A',
    'pos_y':           'N/A',
    'pos_z':           'N/A',
    'roll':            'N/A',
    'pitch':           'N/A',
    'yaw':             'N/A',
    'acc_x':           'N/A',
    'acc_y':           'N/A',
    'acc_z':           'N/A',
    'hdop':            'N/A',
    'wave_prob':       0.0
}
TELEMETRY_LOCK = threading.Lock()

# ---- Load models ----
print("Loading models...")
detector = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
detector.conf = CONF_THRESHOLD
wave_model = tf.keras.models.load_model(WAVE_MODEL_PATH)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print("Models loaded.")

# ---- Initialize webcam ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

roi_buffer = deque(maxlen=CLIP_LENGTH)
latest_prob = 0.0

# ---- Flask app ----
app = Flask(__name__)
HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Drone Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { margin:0; display:flex; height:100vh; font-family:Arial,sans-serif; background:#f0f2f5; }
    .video-container { flex:3; }
    .info-panel { flex:1; padding:20px; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.1); overflow:auto; }
    .info-panel h2 { margin-top:0; }
    .telemetry-item { margin:8px 0; }
    #deliverBtn { margin:12px 0; padding:8px 16px; background:#28a745; border:none; color:#fff; cursor:pointer; border-radius:5px; }
    #statusMsg { margin-top:8px; color:#007bff; }
    img { width:100%; height:auto; display:block; }
    canvas { width:100%; height:200px; margin-top:20px; }
  </style>
</head>
<body>
  <div class="video-container">
    <img src="{{ url_for('video_feed') }}" width="''' + str(FRAME_WIDTH) + '''" height="''' + str(FRAME_HEIGHT) + '''" />
  </div>
  <div class="info-panel">
    <h2>Telemetry</h2>
    <!-- Telemetry items -->
''' + '\n'.join([f'    <div class="telemetry-item"><strong>{key.replace("_"," ").title()}:</strong> <span id="{key}">--</span></div>' for key in TELEMETRY if key != 'wave_prob']) + '''
    <div class="telemetry-item"><strong>Wave Prob:</strong> <span id="wave_prob">0.00</span></div>
    <button id="deliverBtn">Deliver Aid</button>
    <div id="statusMsg"></div>
    <canvas id="odometryChart"></canvas>
  </div>
  <script>
    const ctx = document.getElementById('odometryChart').getContext('2d');
    const odomChart = new Chart(ctx, {
      type: 'line', data: { datasets: [{ data: [], borderColor: 'rgb(54,162,235)', fill: false }] },
      options: { animation: { duration: 0 }, scales: { x: { type: 'linear', title: { display: true, text: 'X (m)' } }, y: { title: { display: true, text: 'Y (m)' } } }, plugins: { legend: { display: false } } }
    });

    async function fetchTelemetry() {
      const res = await fetch('/telemetry');
      const data = await res.json();
      for (const [k, v] of Object.entries(data)) {
        const el = document.getElementById(k);
        if (el) el.textContent = (typeof v === 'number' ? v.toFixed(2) : v);
      }
      const x = parseFloat(data.pos_x), y = parseFloat(data.pos_y);
      if (!isNaN(x) && !isNaN(y)) {
        odomChart.data.datasets[0].data.push({ x, y });
        if (odomChart.data.datasets[0].data.length > 500) odomChart.data.datasets[0].data.shift();
        odomChart.update('none');
      }
    }
    document.getElementById('deliverBtn').onclick = async () => {
      document.getElementById('statusMsg').textContent = 'Initiating delivery...';
      const resp = await fetch('/deliver', { method: 'POST' });
      const j = await resp.json();
      document.getElementById('statusMsg').textContent = j.status;
    };
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
    with TELEMETRY_LOCK:
        TELEMETRY['wave_prob'] = latest_prob
        return jsonify(TELEMETRY)

@app.route('/deliver', methods=['POST'])
def deliver():
    print("[Action] Delivery initiated")
    return jsonify({'status': 'Delivery initiated'}), 200

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(gen_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- Pixhawk thread ----
def find_pixhawk_port():
    for p in serial.tools.list_ports.comports():
        if 'PX4' in p.description or 'Pixhawk' in p.description or 'usbmodem' in p.device:
            return p.device
    return PIXHAWK_PORT

def pixhawk_thread():
    master = mavutil.mavlink_connection(find_pixhawk_port(), baud=PIXHAWK_BAUD)
    master.wait_heartbeat()
    # set intervals
    for msg_id, interval in [
        (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 100000),
        (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU, 100000)
    ]:
        master.mav.command_long_send(master.target_system, master.target_component,
                                     mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                     0, msg_id, interval, 0,0,0,0,0)
    # request streams
    for sid in [
        mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
        mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
        mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA3
    ]:
        for _ in range(3):
            master.mav.request_data_stream_send(master.target_system, master.target_component, sid, 4, 1)
    # receive
    while True:
        msg = master.recv_match(type=['SYS_STATUS','GPS_RAW_INT','GLOBAL_POSITION_INT',
                                      'VFR_HUD','LOCAL_POSITION_NED','ATTITUDE','SCALED_IMU'], blocking=True)
        if not msg: continue
        t = msg.get_type()
        with TELEMETRY_LOCK:
            if t == 'SYS_STATUS':
                TELEMETRY['battery_pct']     = f"{max(msg.battery_remaining,0)}%"
                TELEMETRY['battery_voltage'] = f"{msg.voltage_battery/1000:.2f}V"
                TELEMETRY['battery_current'] = f"{msg.current_battery/100:.2f}A"
            elif t == 'GPS_RAW_INT':
                TELEMETRY['lat']        = f"{msg.lat/1e7:.7f}"
                TELEMETRY['lon']        = f"{msg.lon/1e7:.7f}"
                TELEMETRY['fix_type']   = str(msg.fix_type)
                TELEMETRY['satellites'] = str(msg.satellites_visible)
                TELEMETRY['hdop']       = f"{msg.eph/100:.2f}"
            elif t == 'GLOBAL_POSITION_INT':
                TELEMETRY['altitude_rel'] = f"{msg.relative_alt/1000:.1f}m"
                TELEMETRY['altitude_abs'] = f"{msg.alt/1000:.1f}m"
                TELEMETRY['heading']      = f"{msg.hdg/100:.2f}°"
            elif t == 'VFR_HUD':
                TELEMETRY['speed']    = f"{msg.groundspeed:.1f}m/s"
                TELEMETRY['climb']    = f"{msg.climb:.1f}m/s"
                TELEMETRY['throttle'] = f"{msg.throttle}%"
            elif t == 'LOCAL_POSITION_NED':
                TELEMETRY['pos_x'] = f"{msg.x:.2f}"
                TELEMETRY['pos_y'] = f"{msg.y:.2f}"
                TELEMETRY['pos_z'] = f"{msg.z:.2f}"
            elif t == 'ATTITUDE':
                TELEMETRY['roll']  = f"{msg.roll:.2f}°"
                TELEMETRY['pitch'] = f"{msg.pitch:.2f}°"
                TELEMETRY['yaw']   = f"{msg.yaw:.2f}°"
            elif t == 'SCALED_IMU':
                TELEMETRY['acc_x'] = f"{msg.xacc:.2f}m/s²"
                TELEMETRY['acc_y'] = f"{msg.yacc:.2f}m/s²"
                TELEMETRY['acc_z'] = f"{msg.zacc:.2f}m/s²"

# start thread
threading.Thread(target=pixhawk_thread, daemon=True).start()

# ---- Video / wave detection ----
def detect_person_box(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, size=640).xyxy[0].cpu().numpy()
    persons = [d for d in dets if int(d[5]) == 0 and d[4] >= CONF_THRESHOLD]
    if not persons:
        return None
    x1, y1, x2, y2, _, _ = max(persons, key=lambda d: d[4])
    return int(x1), int(y1), int(x2), int(y2)

def gen_frames():
    global latest_prob
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        box = detect_person_box(frame)
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                roi = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (ROI_SIZE, ROI_SIZE))
                roi_buffer.append(roi)
                if len(roi_buffer) == CLIP_LENGTH:
                    clip = preprocess_input(np.stack(roi_buffer).astype('float32'))
                    latest_prob = float(wave_model.predict(clip[None, ...])[0, 0])
                    roi_buffer.clear()
            label = 'Waving' if latest_prob >= 0.5 else 'Person'
            cv2.putText(frame, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Wave p={latest_prob:.2f}", (10, FRAME_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ret2, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import warnings; warnings.filterwarnings('ignore', category=FutureWarning)
    app.run(host='0.0.0.0', port=3000, threaded=True)
