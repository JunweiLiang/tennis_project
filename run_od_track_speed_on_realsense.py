# coding=utf-8
# simple camera image grabbing exercise, from depth camera like Realsense or Femto Bolt

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import numpy as np
import datetime
import time # for fps compute
from collections import defaultdict

# 1. install realsense-viewer through here:
# https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md
# 2. pip install pyrealsense2
import pyrealsense2 as rs

from utils import image_resize
from utils import print_once

from utils import run_od_on_image
from utils import run_od_track_on_image

# This is a good open-source wrapper
# pip install ultralytics
# see tutorial here
#   1. https://docs.ultralytics.com/models/yolov9/
#   2. https://medium.com/@Mert.A/how-to-use-yolov9-for-object-detection-93598ad88d7d
from ultralytics import YOLO
from ultralytics import YOLOWorld

parser = argparse.ArgumentParser()
parser.add_argument("--det_conf", default=0.5, type=float)
parser.add_argument("--save_to_avi", default=None, help="save the visualization video to a avi file")
# see here for all the available models: https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset
# larger "yolov8x.pt" # latency on RTX 2060: 33.3 ms
# yolov10x 27 ms
# yolov10l 20 ms
# small "yolov9t.pt" # latency on RTX 2060: 11 ms
parser.add_argument("--yolo_model_name", default="yolov10x.pt")
parser.add_argument("--tracker_yaml", default="bytetrack.yaml")
parser.add_argument("--use_open_model", action="store_true")
parser.add_argument("--det_only", action="store_true")

# for each track, get the latest 3D point and the last 3D points
x_l, x_r, y_l, y_r = 100, 1280 - 100, 50, 720 - 50

def est_speed_on_tracks(track_history, depth_data, depth_intrin, track_speed_history):
    # this is for realsense
    # for each track, get the latest 3D point and the last 3D points
    global x_l, x_r, y_l, y_r
    for track_id in track_history:
        track = track_history[track_id]
        # excluding any box around the edges, where depth is not good
        track = [x for x in track if x_l < x[0] and x[0] < x_r and y_l < x[1] and x[1] < y_r]
        if len(track) > 1:
            # integers coordinates
            current_x, current_y, cls_id, current_timestamp = track[-1]
            last_x, last_y, _, last_timestamp = track[-2]

            # in meters
            current_depth = depth_data[current_y, current_x]
            last_depth =depth_data[last_y, last_x]

            current_point3d = rs.rs2_deproject_pixel_to_point(
                depth_intrin,
                (current_x, current_y),
                current_depth)
            last_point3d = rs.rs2_deproject_pixel_to_point(
                depth_intrin,
                (last_x, last_y),
                last_depth)

            dist = np.linalg.norm(np.array(current_point3d) - np.array(last_point3d))
            speed = dist / (current_timestamp - last_timestamp) # meters / second

            track_speed = track_speed_history[track_id]
            track_speed.append(speed)
            if len(track_speed) > 3000:
                track_speed.pop(0)


if __name__ == "__main__":
    args = parser.parse_args()

    # load the model first
    # initialize the object detection model

    detection_classes = [32] #  0 person, 32 sports ball on COCO
    if args.use_open_model:
        detection_classes = ["person", "tennis ball"] # does not work yet
        # https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt
        model = YOLOWorld("yolov8x-worldv2.pt")
        model.set_classes(detection_classes)
    else:
        # tranditional COCO detection model
        # this will auto download the YOLOv9 checkpoint
        # see here for all the available models: https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset
        model = YOLO(args.yolo_model_name)



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
        track_history = defaultdict(list)
        track_speed_history = defaultdict(list)

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
            current_time = time.time()
            fps = int(frame_count / (current_time - start_time))

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # mm to meters
            depth_data = np.asanyarray(depth_frame.get_data()) * depth_scale
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

            # see here for inference arguments
            # https://docs.ultralytics.com/modes/predict/#inference-arguments
            if args.det_only:
                color_image, det_results = run_od_on_image(
                        color_image, model, classes=detection_classes, conf=args.det_conf,
                        bbox_thickness=4) # larger box to be overwritten by track results

                # save the data into a single track, assume only object in the scene (like a tennis ball)
                class_for_speed_est = 32  # sports ball
                # xy is the center point coordinate
                boxes = [box.xywh[0].cpu() for box in det_results[0].boxes if box.cls[0] == class_for_speed_est]
                if len(boxes) > 0:
                    boxes_and_area = [(xywh, xywh[2]*xywh[3]) for xywh in boxes]
                    boxes_and_area.sort(reverse=True, key=lambda x: x[1])
                    # only keeping the largest one
                    center_x, center_y, _, _ = boxes_and_area[0][0]
                    current_timestamp = time.time()
                    track_history["tennis ball"].append((int(center_x), int(center_y), 0, current_timestamp))

            else:

                color_image, track_results = run_od_track_on_image(
                    color_image, model, track_history,
                    classes=detection_classes, conf=args.det_conf,
                    tracker_yaml=args.tracker_yaml)

                result = track_results[0]

            # get track_id -> a list of speed, the last one is the latest speed
            est_speed_on_tracks(
                track_history, depth_data, depth_intrin,
                track_speed_history)

            # print out the speed on the image (trackid, current speed, max speed, mean speed)
            speed_to_print = [
                    #(track_id, np.mean(speeds[-30:-1]), np.percentile(speeds, 95), np.mean(speeds))
                    (track_id, np.mean(speeds[-fps:]), np.max(speeds[-fps*3:]), np.mean(speeds[-fps*30:]))
                    for track_id, speeds in track_speed_history.items()]
            speed_to_print.sort(key=lambda x: x[0])

            image = color_image

            # draw the area we will be estimating speed
            image = cv2.rectangle(image, (x_l, y_l), (x_r, y_r), (0, 255, 0), 2)

            unit = "m/s"
            start_bottom_y = 680
            end_bottom_y = 680 - len(speed_to_print)*20
            image = cv2.rectangle(image, (0, end_bottom_y-1), (1280, start_bottom_y), (0, 0, 0), -1)
            for i, (track_id, current_s, max_s, mean_s) in enumerate(speed_to_print):
                if type(track_id) is str:
                    track_name = track_id
                else:
                    track_name = "%s #%d" % (result.names[track_history[track_id][0][2]], track_id)
                image = cv2.putText(
                    image, "%s: speed in last 1s %.1f, max %.1f in last 3s, avg. %.1f %s in last 30s" % (
                        track_name, current_s, max_s, mean_s, unit),
                    (10, start_bottom_y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 255, 0), thickness=2)
                start_bottom_y -= 10

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

            image = cv2.putText(
                image, "FPS: %d" % int(fps),
                (10, 710), cv2.FONT_HERSHEY_SIMPLEX,
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
