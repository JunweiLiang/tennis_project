# coding=utf-8
# simple camera image grabbing exercise, from depth camera like Realsense or Femto Bolt

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--camera_type", default="realsense")

# 1. example use to get an image from the laptop camera
# (base) junweil@precognition-laptop2:~$ python ~/projects/tennis_project/get_camera_image.py Downloads/output.png
# if you see
#   [ WARN:0@0.012] global cap_v4l.cpp:999 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
# 可能需要在笔记本电脑上登录一下你的账号，唤醒一下摄像头


def show_point_depth(point, depth_image, color_image):
    """
        point: (y, x)
    """
    depth = depth_image[point[0], point[1]]  # in milimeters
    color_image = cv2.circle(
        color_image,
        (point[1], point[0]), radius=2, color=(0, 255, 0), thickness=2)
    color_image = cv2.putText(
        color_image, "d: %smm" % depth,
        (point[1] - 10, point[0]), cv2.FONT_HERSHEY_SIMPLEX,
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
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # depth_value * depth_scale -> meters
        print("Depth Scale is: " , depth_scale)
        print("aligning depth frame to RGB frames..") # depth sensor has different extrinsics with RGB sensor
        align_to = rs.stream.color
        aligner = rs.align(align_to)

    else:
        print("not supported camera.")
        sys.exit()


    print("Now showing the camera stream. press Q to exit.")
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # 1. we need to align the frames, so on the x,y of RGB, we get the correct depth
            aligned_frames = aligner.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # junwei: the color_intrin and depth_intrin are the same as they are aligned.
            #### 获取相机参数 ####
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
            aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

            #print(depth_intrin, color_intrin)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # (480, 640, 3), (480, 640)
            # depth_image are in meters
            #print(color_image.shape, depth_image.shape)
            #print(depth_image[240, 320]) # 单位：毫米

            # showing two points' depth
            point1 = (150, 320)
            point2 = (240, 320)

            color_image, depth1 = show_point_depth(point1, depth_image, color_image)
            color_image, depth2 = show_point_depth(point2, depth_image, color_image)

            # rs2_deproject_pixel_to_point takes pixel (x, y)
            # outputs (x, y, z), the coordinates are in meters
            point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, (point1[1], point1[0]), depth1)
            point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, (point2[1], point2[0]), depth1)
            print(point1_3d, point2_3d)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            image = np.hstack((color_image, depth_colormap))

            # Show the image
            cv2.imshow('RGB and Depth Stream', image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
