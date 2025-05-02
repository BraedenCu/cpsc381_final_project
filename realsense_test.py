#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1) Configure and start the RealSense pipeline
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            # 2) Wait for a coherent pair of frames: here only color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 3) Convert to NumPy array and show
            img = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Test', img)

            # 4) Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 5) Stop streaming and close windows
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()