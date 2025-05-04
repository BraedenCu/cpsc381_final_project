# jetson_publisher.py
# ------------------------------------------
# Captures camera frames and MAVLink odometry,
# then POSTs both to your MacBook.

import time
import json
import cv2
import requests
from pymavlink import mavutil

# 1) Configure MAVLink over CDC-ACM
mav = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)

# 2) Open the first USB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera")

# 3) Destination on your Mac (replace with your Mac’s IP)
SERVER_URL = 'http://<MAC_IP>:5000/receive'

while True:
    # a) Grab a frame
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed, retrying…")
        time.sleep(0.1)
        continue

    # b) Pull the latest GLOBAL_POSITION_INT message
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
    if msg:
        odom = {
            'lat': msg.lat / 1e7,
            'lon': msg.lon / 1e7,
            'alt': msg.alt / 1000.0
        }
    else:
        odom = {}

    # c) JPEG-encode the frame in memory
    _, buf = cv2.imencode('.jpg', frame)
    files = {
        'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')
    }
    data = {
        'odometry': json.dumps(odom)
    }

    # d) POST to Mac
    try:
        resp = requests.post(SERVER_URL, files=files, data=data, timeout=1)
        if resp.status_code != 200:
            print(f"Server error: {resp.status_code}")
    except requests.RequestException as e:
        print(f"Failed to send: {e}")

    time.sleep(0.1)