# coding=utf-8
# simple camera image grabbing exercise, from depth camera like Realsense or Femto Bolt

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import numpy as np
import datetime
import time # for fps compute

from utils import image_resize
from utils import print_once

parser = argparse.ArgumentParser()

parser.add_argument("--camera_type", default="realsense")
parser.add_argument("--save_to_avi", default=None, help="save the visualization/rgb video to a avi file")
parser.add_argument("--save_data_only", action="store_true", help="if true, the saved video only contains RGB stream")
parser.add_argument("--depth_data_file", default=None, help="save the depth data here")


# 1. example use to get an image from the laptop camera
# (base) junweil@precognition-laptop2:~$ python ~/projects/tennis_project/get_depth_images.py --camera_type realsense


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


if __name__ == "__main__":
    args = parser.parse_args()

    # Assume we are connected to a depth camera
    # lazy import API for specified depth camera
    if args.camera_type == "realsense":

        # 1. install realsense-viewer through here:
        # https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md
        # 2. pip install pyrealsense2
        import pyrealsense2 as rs

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

    else:
        print("not supported camera.")
        sys.exit()


    print("Now showing the camera stream. press Q to exit.")
    start_time = time.time()
    frame_count = 0
    depth_data_dict = {}
    try:
        if args.save_to_avi is not None:

            # cannot save to mp4 file, due to liscensing problem, need to compile opencv from source
            print("saving to avi video %s..." % args.save_to_avi)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            width_height = (1920, 1080)
            if args.save_data_only:
                # only saving the RGB video
                width_height = (1280, 720)
            out = cv2.VideoWriter(args.save_to_avi, fourcc, 30.0, width_height)

        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # 1. we need to align the frames, so on the x,y of RGB, we get the correct depth
            aligned_frames = aligner.process(frames)
            aligned_frames.keep()  # realsense's problem
            # https://support.intelrealsense.com/hc/en-us/community/posts/4410630729619-RuntimeError-Error-occured-during-execution-of-the-processing-block-See-the-log-for-more-info

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # frame_count start from 1
            frame_count += 1

            # junwei: the color_intrin and depth_intrin are the same as they are aligned.
            #### 获取相机参数 ####
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
            aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            # [ 640x480  p[325.217 238.38]  f[385.38 384.848]  Inverse Brown Conrady [-0.0565123 0.067672 0.000208852 0.000719325 -0.0218305] ]
            color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

            #print(depth_intrin, color_intrin)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # (480, 640, 3), (480, 640)
            # depth_image are in meters
            #print(color_image.shape, depth_image.shape)
            #print(depth_image[240, 320]) # 单位：毫米

            if args.save_data_only:
                image = color_image
                depth_data_int_array = depth_image.astype(np.int16)
                depth_data_dict[frame_count] = depth_data_int_array
            else:
                # for visualization

                # showing two points' depth
                point1 = (400, 400)  # (y, x)
                point2 = (480, 640)

                color_image, depth1 = show_point_depth(point1, depth_image, color_image)
                color_image, depth2 = show_point_depth(point2, depth_image, color_image)
                # mm to meters
                depth1 = depth1 * depth_scale
                depth2 = depth2 * depth_scale

                # rs2_deproject_pixel_to_point takes pixel (x, y)
                # outputs (x, y, z), the coordinates are in meters
                #   [0,0,0] is the center of the camera, 相机朝向的右边是正x，下边为正y, 朝向是正z
                #   See this doc for coordinate system
                #   https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0?fbclid=IwAR3gogVZe824YUps88Dzp02AN_XzEm1BDb0UbmzfoYvn1qDFb7KzbIz9twU#point-coordinates
                # 理解此函数，需要知道camera model，perspective projection, geometric computer vision
                # 也就是说3D世界的坐标如何与相机上的像素坐标互相转换的
                point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, (point1[1], point1[0]), depth1)
                point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, (point2[1], point2[0]), depth2)

                # 计算这两点的实际距离
                #print(point1_3d, point2_3d)
                dist_between_point1_point2 = np.linalg.norm(np.array(point1_3d) - np.array(point2_3d))

                mid_point_xy = ( int((point2[1] + point1[1])/2.), int((point2[0] + point1[0])/2.) + 100)
                color_image = cv2.putText(
                    color_image, "dist 1to2: %.2f meters" % dist_between_point1_point2,
                    mid_point_xy, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                image = np.hstack((color_image, depth_colormap))

                image = image_resize(image, width=1920, height=None)

                print_once("image shape: %s" % list(image.shape[:2]))


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
        if args.depth_data_file is not None:
            np.savez_compressed(args.depth_data_file, depth_data_dict)
        cv2.destroyAllWindows()
