#!/usr/bin/env python3
"""
Entire script for first detecting if there is a person and then if the person is waving 
using the RealSense
First, single-frame detection using YOLOv5m
Then, create an 8-frame rolling buffer of cropped ROIs
Then, do waving inference with MobileNetV2+LSTM
Last, depth → 3D deprojection 

RIGHT NOW DOES NOT ACTUALLY MOVE THE DRONE, J PRINTS OUT WHERE THE PERSON WAVING IS 
"""


import pyrealsense2 as rs
import torch
import tensorflow as tf
import cv2
import numpy as np
from collections import deque

YOLO_MODEL_PATH    = 'weights/yolov5m.pt' # pre-trained YOLOv5 model
WAVE_MODEL_PATH    = 'weights/wave_sequence_model_one_epoch.h5' # our own trained waving model
CONF_THRESHOLD     = 0.5 # person detection confidence
FRAME_WIDTH        = 640  # RealSense frame width
FRAME_HEIGHT       = 480  # RealSense frame height
CLIP_LENGTH        = 8  # The number of frames per clip
ROI_SIZE           = 224  # crop size from YOLO -> Wave because Wave takes in 224x224 clips
# Roi -> region of interest

# loads the YOLOv5 person detector and waving sequence model
def load_models():
    # load YOLOv5 custom (COCO) for person class
    detector = torch.hub.load(
        'ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, source='local'
    )
    detector.conf = CONF_THRESHOLD

    # Load  keras waving model
    wave_model = tf.keras.models.load_model(WAVE_MODEL_PATH)
    return detector, wave_model

# init RealSense pipeline and alignment.
def init_realsense():
    pipeline = rs.pipeline()


    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)



    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_intrin = depth_stream.as_video_stream_profile().get_intrinsics()
    return pipeline, align, depth_intrin


# run the YOLOv5 person detector on a BGR frame --> returns (x1,y1,x2,y2) of highest-confidence person or None 
def detect_person_box(detector, frame):


    # YOLOv5 expects RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector(img, size=640)  
    dets = results.xyxy[0].cpu().numpy()  # [N,6]: x1,y1,x2,y2,conf,class



    # filter for COCO class 0 (which is person)
    persons = [d for d in dets if int(d[5]) == 0 and d[4] >= CONF_THRESHOLD]

    if not persons:
        return None
    # pick highest-confidence

    x1, y1, x2, y2, conf, cls = max(persons, key=lambda d: d[4])
    return int(x1), int(y1), int(x2), int(y2)
381

def main():
    # Load models
    person_detector, wave_model = load_models()
    print("models loaded")


    # Initialize RealSense
    pipeline, align, depth_intrin = init_realsense()
    print("RealSense init done.")

    # Rolling buffer for ROI frames

    roi_buffer = deque(maxlen=CLIP_LENGTH)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    try:
        while True:
            # 1) get aligned frames
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)


            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue



            color_img = np.asanyarray(color_frame.get_data())

            # 2) detect person box
            box = detect_person_box(person_detector, color_img)

            if box is None:

                print("no person detected")
                roi_buffer.clear()
                continue

            x1, y1, x2, y2 = box


            # 3) Crop & resize ROI
            crop = color_img[y1:y2, x1:x2]
            if crop.size == 0:
                print("Empty crop, skipping.")
                roi_buffer.clear()
                continue

            roi = cv2.resize(crop, (ROI_SIZE, ROI_SIZE))
            roi_buffer.append(roi)
            print(f"Buffered {len(roi_buffer)}/{CLIP_LENGTH} ROI frames.")

            # 4) waving inference once buffer is full
            if len(roi_buffer) == CLIP_LENGTH:
                clip = np.stack(roi_buffer, axis=0).astype('float32')
                clip = preprocess_input(clip)

                # model expects batch dim
                prob = wave_model.predict(clip[None, ...])[0, 0]
                waving = prob >= 0.5
                print(f"Waving prob={prob:.2f} → {waving}")

                if waving:

                    # 5) get 3D coordinates from center pixel
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    depth = depth_frame.get_distance(cx, cy)
                    point = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [cx, cy], depth
                    )


                    print(f"Detected wave at 3D point (m): {point}")
                    # IF WE HAVE THE TIME TO DO IT  replace with drone control call

                # Clear buffer to await next wave
                roi_buffer.clear()

    finally:
        pipeline.stop()
        print("Pipeline stopped.")


if __name__ == '__main__':
    main()
