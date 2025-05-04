# jetson_streamer.py
#!/usr/bin/env python3
import os
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, stream_with_context
from pymavlink import mavutil
import serial.tools.list_ports

# ---- Configuration ----
PIXHAWK_BAUD   = 57600
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
SERVER_PORT    = 3000

# ---- Shared telemetry state ----
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
    'hdop':            'N/A'
}
TEL_LOCK = threading.Lock()

# ---- Open camera ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open USB camera")

# ---- Flask setup ----
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(
        stream_with_context(gen_frames()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/telemetry')
def telemetry():
    with TEL_LOCK:
        return jsonify(TELEMETRY)

def find_pixhawk_port():
    for p in serial.tools.list_ports.comports():
        if 'PX4' in p.description or 'Pixhawk' in p.description or 'usbmodem' in p.device:
            return p.device
    return None

# ---- Background thread to poll Pixhawk ----
def pixhawk_thread():
    port = find_pixhawk_port()
    if port is None:
        print("Warning: could not auto-detect Pixhawk port, using /dev/ttyACM0")
        port = '/dev/ttyACM0'
    master = mavutil.mavlink_connection(port, baud=PIXHAWK_BAUD)
    master.wait_heartbeat()
    # request the common streams once
    for msg_id, us in [
        (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,       500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,      500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,          500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,         200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU,       200000)
    ]:
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, msg_id, us, 0,0,0,0,0
        )
    # loop forever
    while True:
        msg = master.recv_match(
            type=[
                'SYS_STATUS','GPS_RAW_INT','GLOBAL_POSITION_INT',
                'VFR_HUD','LOCAL_POSITION_NED','ATTITUDE','SCALED_IMU'
            ], blocking=True
        )
        if not msg:
            continue
        t = msg.get_type()
        with TEL_LOCK:
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
                TELEMETRY['altitude_abs'] = f"{msg.alt/1000:.1f}m"
                TELEMETRY['altitude_rel'] = f"{msg.relative_alt/1000:.1f}m"
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

# ---- Frame generator for MJPEG ----
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes +
               b'\r\n')

if __name__ == '__main__':
    # start Pixhawk polling
    threading.Thread(target=pixhawk_thread, daemon=True).start()

    # serve HTTP
    try:
        app.run(host='0.0.0.0', port=SERVER_PORT, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("🛑 Camera released, exiting.")