#!/usr/bin/env python3
import os
import threading
import time
import cv2
import numpy as np
import serial.tools.list_ports
import serial
from flask import Flask, Response, jsonify, stream_with_context
from pymavlink import mavutil

# ---- Configuration ----
PIXHAWK_BAUD   = 57600
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
SERVER_PORT    = 3000
ARDUINO_BAUD   = 115200

# ---- Shared telemetry state & lock ----
TELEMETRY = { k: 'N/A' for k in [
    'battery_pct','battery_voltage','battery_current',
    'throttle','speed','climb','altitude_rel','altitude_abs',
    'heading','lat','lon','fix_type','satellites',
    'pos_x','pos_y','pos_z','roll','pitch','yaw',
    'acc_x','acc_y','acc_z','hdop'
]}
TELE_LOCK = threading.Lock()

# ---- Camera setup ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# ---- Arduino setup (dropper) ----
def find_arduino_port():
    """Scan for an Arduino-style USB-serial by manufacturer or ttyACM."""
    for p in serial.tools.list_ports.comports():
        if 'arduino' in (p.manufacturer or '').lower() or 'ttyacm' in p.device.lower():
            return p.device
    return None

arduino_ser = None
arduino_port = find_arduino_port()
if arduino_port:
    try:
        arduino_ser = serial.Serial(arduino_port, ARDUINO_BAUD, timeout=1)
        print(f"Arduino connected on {arduino_port}")
    except Exception as e:
        print(f"Failed to open Arduino on {arduino_port}: {e}")
else:
    print("Arduino not found")

# ---- Pixhawk port detection & telemetry thread ----
def find_pixhawk_port():
    """
    Continuously scan all serial ports for a Pixhawk device.
    Matches on 'px4' or 'pixhawk' in description, USB VID/PID,
    or device names like /dev/ttyACM* or /dev/ttyUSB* (excluding Arduino).
    """
    while True:
        ports = list(serial.tools.list_ports.comports())  # :contentReference[oaicite:3]{index=3}
        # 1) look for PX4/Pixhawk descriptions or known VID/PID in hwid
        for p in ports:
            desc = (p.description or '').lower()
            hwid = (p.hwid or '').lower()
            if 'px4' in desc or 'pixhawk' in desc:
                return p.device
            if '26ac' in hwid:  # common PX4 VID :contentReference[oaicite:4]{index=4}
                return p.device
        # 2) fallback: any ttyACM or ttyUSB not the Arduino port
        for p in ports:
            dev = p.device.lower()
            if (dev.startswith('/dev/ttyacm') or dev.startswith('/dev/ttyusb')) and p.device != arduino_port:
                return p.device
        # 3) no port yet—wait and retry
        print("No Pixhawk port found, retrying in 2s…")
        time.sleep(2)

def pixhawk_thread():
    # wait until we successfully open the Pixhawk link
    while True:
        port = find_pixhawk_port()
        print(f"Attempting Pixhawk connection on {port}")
        try:
            master = mavutil.mavlink_connection(port, baud=PIXHAWK_BAUD)  # :contentReference[oaicite:5]{index=5}
            break
        except Exception as e:
            print(f"Failed to connect on {port}: {e}")
            time.sleep(2)
    print("Heartbeat received from Pixhawk")
    master.wait_heartbeat()

    # request common data streams at 4 Hz
    for stream_id in [
        mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
        mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
        mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA3
    ]:
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            stream_id,
            4,  # 4 Hz
            1   # start streaming :contentReference[oaicite:6]{index=6}
        )

    # set higher-rate intervals for key messages
    for msg_id, interval_us in [
        (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,         500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,        500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,            200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,           200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU,         200000)
    ]:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0,0,0,0,0  # :contentReference[oaicite:7]{index=7}
        )

    # continuously read and store telemetry
    while True:
        msg = master.recv_match(blocking=True)
        if not msg:
            continue
        t = msg.get_type()
        with TELE_LOCK:
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

# ---- MJPEG generator ----
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# ---- Flask app & endpoints ----
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(gen_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    with TELE_LOCK:
        return jsonify(TELEMETRY)

@app.route('/drop', methods=['POST'])
def drop():
    if not arduino_ser:
        return jsonify({'status':'Arduino not connected'}), 500
    try:
        arduino_ser.write(b"DROP\n")
        return jsonify({'status':'Drop sent'}), 200
    except Exception as e:
        return jsonify({'status':'Error','error':str(e)}), 500

# ---- Main ----
if __name__ == '__main__':
    threading.Thread(target=pixhawk_thread, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=SERVER_PORT, threaded=True)
    finally:
        cap.release()
        if arduino_ser:
            arduino_ser.close()
        print("Clean exit")
        os._exit(0)
