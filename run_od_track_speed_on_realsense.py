# coding=utf-8
# simple camera image grabbing exercise, from depth camera like Realsense or Femto Bolt

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import numpy as np
import datetime
import time # for fps compute

# 1. install realsense-viewer through here:
# https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md
# 2. pip install pyrealsense2
import pyrealsense2 as rs

from utils import image_resize
from utils import print_once
from collections import defaultdict

# This is a good open-source wrapper
# pip install ultralytics
# see tutorial here
#   1. https://docs.ultralytics.com/models/yolov9/
#   2. https://medium.com/@Mert.A/how-to-use-yolov9-for-object-detection-93598ad88d7d
from ultralytics import YOLO

parser = argparse.ArgumentParser()

parser.add_argument("--save_to_avi", default=None, help="save the visualization video to a avi file")


def show_point_depth(point, depth_image, color_image):
    """
        point: (y, x)
    """
    depth = depth_image[point[0], point[1]]  # in milimeters
    color_image = cv2.circle(
        color_image,
        (point[1], point[0]), radius=2, color=(0, 255, 0), thickness=2)
    color_image = cv2.putText(
        color_image, "depth: %dmm" % int(depth),
        (point[1], point[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, color=(0, 255, 0), thickness=2)
    return color_image, depth


def run_od_on_image(
        frame_cv2, od_model,
        classes=[], conf=0.5,
        bbox_thickness=4, text_thickness=2, font_size=2):
    """
        run object detection inference and visualize in the image
    """

    # see here for inference arguments
    # https://docs.ultralytics.com/modes/predict/#inference-arguments
    results = od_model.predict(
        frame_cv2,
        classes=None if len(classes)==0 else classes,  # you can specify the classes you want
        # see here for coco class indexes [0-79], 0 is person: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
        #classes=[0, 32], # detect person and sports ball only
        conf=conf,
        #half=True
        )

    # see here for the API documentation of results
    # https://docs.ultralytics.com/modes/predict/#working-with-results
    result = results[0] # we run it on single image
    for box in result.boxes:
        bbox = [int(x) for x in box.xyxy[0]]
        bbox_color = (0, 255, 0) # BGR
        frame_cv2 = cv2.rectangle(
                frame_cv2,
                tuple(bbox[0:2]), tuple(bbox[2:4]),
                bbox_color, bbox_thickness)

        frame_cv2 = cv2.putText(
                frame_cv2, "%s" % result.names[int(box.cls[0])],
                (bbox[0], bbox[1] - 10),  # specify the bottom left corner
                cv2.FONT_HERSHEY_PLAIN, font_size,
                bbox_color, text_thickness)
    return frame_cv2, results

def run_od_track_on_image(
        frame_cv2, od_model, track_history,
        classes=[], conf=0.5,
        bbox_thickness=4, text_thickness=2, font_size=2):
    """
        run object detection and tracking on a new frame, and visualize
    """

    # see here for inference arguments
    # https://docs.ultralytics.com/modes/track/#tracking
    results = od_model.track(
        frame_cv2,
        #tracker="bytetrack.yaml",
        tracker="botsort.yaml",
        classes=None if len(classes)==0 else classes,  # you can specify the classes you want
        # see here for coco class indexes [0-79], 0 is person: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
        #classes=[0, 32], # detect person and sports ball only
        conf=conf,
        iou=0.5,
        persist=True
        )

    # see here for the API documentation of results
    # https://docs.ultralytics.com/modes/predict/#working-with-results
    result = results[0]

    # Get the boxes and track IDs for ploting the lines
    boxes = result.boxes.xywh.cpu()
    boxes_xyxy = result.boxes.xyxy.cpu()
    track_ids = result.boxes.id.int().cpu().tolist()
    classes = result.boxes.cls.int().cpu().tolist()

    for box, box_xyxy, track_id, cls_id in zip(boxes, boxes_xyxy, track_ids, classes):
        center_x, center_y, w, h = box
        x1, y1, x2, y2 = box_xyxy

        track = track_history[track_id]

        bbox_color = (255, 0, 0) # BGR

        frame_cv2 = cv2.rectangle(
                frame_cv2,
                (int(x1), int(y1)), (int(x2), int(y2)),
                bbox_color, bbox_thickness)

        frame_cv2 = cv2.putText(
                frame_cv2, "%s #%d" % (result.names[cls_id], track_id),
                (int(x1), int(y1) - 10),  # specify the bottom left corner
                cv2.FONT_HERSHEY_PLAIN, font_size,
                bbox_color, text_thickness)
    return frame_cv2, results


if __name__ == "__main__":
    args = parser.parse_args()

    # load the model first
    # initialize the object detection model
    # this will auto download the YOLOv9 checkpoint
    # see here for all the available models: https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset
    model = YOLO("yolov9t.pt")

    # Configure RealSense pipeline for depth and RGB.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()

    # depth_value * depth_scale -> meters
    depth_scale = depth_sensor.get_depth_scale()  # 0.001

    print("Depth Scale is: " , depth_scale)
    print("aligning depth frame to RGB frames..") # depth sensor has different extrinsics with RGB sensor
    align_to = rs.stream.color
    aligner = rs.align(align_to)


    print("Now showing the camera stream. press Q to exit.")
    start_time = time.time()
    frame_count = 0
    depth_data_dict = {}
    try:
        if args.save_to_avi is not None:

            # cannot save to mp4 file, due to liscensing problem, need to compile opencv from source
            print("saving to avi video %s..." % args.save_to_avi)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")

            # the visualization video size
            width_height = (1280, 720)
            out = cv2.VideoWriter(args.save_to_avi, fourcc, 30.0, width_height)


        # Store the track history
        track_history = defaultdict(lambda: [])

        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # 1. we need to align the frames, so on the x,y of RGB, we get the correct depth
            aligned_frames = aligner.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # frame_count start from 1
            frame_count += 1


            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # (720, 1280, 3), (720, 1280)
            #print(color_image.shape, depth_image.shape)
            #print(depth_image[240, 320]) # 单位：毫米

            # see here for inference arguments
            # https://docs.ultralytics.com/modes/predict/#inference-arguments
            #color_image, _ = run_od_on_image(color_image, model, classes=[0, 32])
            color_image, _ = run_od_track_on_image(color_image, model, track_history, classes=[0, 32])

            image = color_image

            # put a timestamp for the frame for possible synchronization
            # and a frame index to look up depth data
            date_time = str(datetime.datetime.now())
            image = cv2.putText(
                image, "#%d: %s" % (frame_count, date_time),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)

            if args.save_to_avi is not None:

                out.write(image)

            # show the fps in the visualization
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            image = cv2.putText(
                image, "FPS: %d" % int(fps),
                (10, 700), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)

            # Show the image
            cv2.imshow('RGB and Depth Stream', image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        pipeline.stop()
        if args.save_to_avi is not None:
            out.release()
        cv2.destroyAllWindows()
