import threading
import time
import numpy as np
import pyrealsense2 as rs
import cv2
from flask import Flask, Response, render_template_string

# --- Camera thread that keeps latest frame ---
class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(cfg)
        self.frame = None
        self.lock  = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            img = np.array(color.get_data())
            with self.lock:
                self.frame = img

    def get_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, buf = cv2.imencode('.jpg', self.frame)
            return buf.tobytes() if ret else None

cam = Camera()
app = Flask(__name__)

@app.route('/')
def index():
    # Simple page with <img> tag for our stream
    return render_template_string("""
    <html><body>
      <h1>RealSense D435 Live</h1>
      <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body></html>
    """)

def mjpeg_generator():
    while True:
        jpeg = cam.get_jpeg()
        if jpeg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/stream')
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # Expose on all interfaces so other devices can connect
    app.run(host='0.0.0.0', port=5000, threaded=True)
    # Cleanup on exit
    cam.pipeline.stop()
    cam.pipeline = None
    cam.frame = None
    cam.lock = None
    cv2.destroyAllWindows()