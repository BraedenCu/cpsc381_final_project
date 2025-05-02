#!/usr/bin/env python3
import threading
import time
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string

# --- Global state for the latest center distance ---
current_distance = 0.0

class CenterDistanceCamera:
    def __init__(self):
        # Configure pipeline for color + depth
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # Start streaming
        self.pipeline.start(cfg)
        # Intrinsics for depth stream (to compute center coords)
        profile = self.pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs.stream.depth)
        intr = depth_stream.as_video_stream_profile().intrinsics
        self.width = intr.width
        self.height = intr.height

    def update_distance(self):
        global current_distance
        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            # Center pixel coordinates
            cx = self.width  // 2
            cy = self.height // 2
            # Get center distance in meters
            dist = depth_frame.get_distance(cx, cy)  # get_distance(x,y) → meters  [oai_citation:3‡Stack Overflow](https://stackoverflow.com/questions/52677479/exact-depth-distance-from-realsense-d435-with-x-y-coordinates?utm_source=chatgpt.com)
            current_distance = dist
            time.sleep(0.05)  # ~20 Hz update

    def stop(self):
        self.pipeline.stop()

# Start camera thread
cam = CenterDistanceCamera()
threading.Thread(target=cam.update_distance, daemon=True).start()

# --- Flask setup ---
app = Flask(__name__)

# HTML page with JavaScript SSE subscription
HTML = """
<!doctype html>
<html>
  <head><title>Center Distance</title></head>
  <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
    <h1>RealSense Center Distance</h1>
    <div id="distance" style="font-size: 2em;">-- m</div>
    <script>
      const evtSource = new EventSource("/distance_stream");
      const el = document.getElementById("distance");
      evtSource.onmessage = e => {
        el.textContent = parseFloat(e.data).toFixed(3) + " m";
      };
    </script>
  </body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

def distance_event_stream():
    # SSE: stream lines like "data: <value>\n\n"
    while True:
        yield f"data: {current_distance}\n\n"
        time.sleep(0.1)  # 10 Hz stream

@app.route('/distance_stream')
def distance_stream():
    return Response(distance_event_stream(),
                    mimetype='text/event-stream')  # SSE mimetype  [oai_citation:4‡GitHub](https://github.com/MaxHalford/flask-sse-no-deps?utm_source=chatgpt.com)

if __name__ == '__main__':
    try:
        # Host on all interfaces so clients can connect remotely
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        cam.stop()