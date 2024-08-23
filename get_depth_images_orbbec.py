# coding=utf-8
# simple camera image grabbing exercise, from depth camera like Orbbec Femto Bolt
# https://github.com/orbbec/pyorbbecsdk/blob/main/docs/README_EN.md

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import numpy as np
import time
from utils import image_resize

parser = argparse.ArgumentParser()

parser.add_argument("--camera_type", default="orbbec")

# 1. example use to get an image from the laptop camera
# (base) junweil@precognition-laptop2:~$ python ~/projects/tennis_project/get_depth_images.py --camera_type realsense


def show_point_depth(point, depth_image, color_image):
    """
        point: (y, x)
    """
    depth = depth_image[point[0], point[1]]  # in milimeters
    color_image = cv2.circle(
        color_image,
        (point[1], point[0]), radius=2, color=(0, 255, 0), thickness=4)
    color_image = cv2.putText(
        color_image, "depth: %.3fm" % depth,
        (point[1], point[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2, color=(0, 255, 0), thickness=4)
    return color_image, depth

def get_orbbec_depth_data(orbbec_depth_frame):
    width = orbbec_depth_frame.get_width()
    height = orbbec_depth_frame.get_height()
    scale = orbbec_depth_frame.get_depth_scale() # scale is 1.0, the HW are in milimeters
    scale = 0.001 # I want them in meters

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    depth_data = depth_data.astype(np.float32) * scale

    return depth_data


def get_orbbec_color_data(orbbec_color_frame):
    # in BGR order
    width = orbbec_color_frame.get_width()
    height = orbbec_color_frame.get_height()
    color_format = orbbec_color_frame.get_format()

    assert color_format in [OBFormat.RGB]

    if color_format == OBFormat.RGB:
        data = np.asanyarray(orbbec_color_frame.get_data())
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def deproject_pixel_to_point(camera_param, xy, depth):
    # this is the opposite process of perspective project (3D to 2D), hence "deproject"
    # see slide 23-36 from AIAA 5036 Lecture 4
    #   https://hkust-aiaa5036.github.io/spring2024/lecs.html
    # remember the key formulation: Pc = K Rt Pw (Pc is the pixel coor (homogenous), Pw is the world coor)
    # so Pw = Pc dot K (transposed), supposing we dont care about extrinsics
    # assuming depth is in meters
    x, y = xy
    fx, fy = camera_param.rgb_intrinsic.fx, camera_param.rgb_intrinsic.fy
    cx, cy = camera_param.rgb_intrinsic.cx, camera_param.rgb_intrinsic.cy

    # See Slide 28
    # see also https://stackoverflow.com/questions/38909696/2d-coordinate-to-3d-world-coordinate/38914061#38914061
    x_3d = depth / fx * (x - cx)
    y_3d = depth / fy * (y - cy)

    return [x_3d, y_3d, depth]

def deproject_pixel_to_point_matmul(camera_param, xy, depth):
    intrinsic = np.identity(3) # [1 0 0][0 1 0][0 0 1]
    # focal length
    intrinsic[0, 0] = camera_param.rgb_intrinsic.fx
    intrinsic[1, 1] = camera_param.rgb_intrinsic.fy
    # center of image
    intrinsic[0, 2] = camera_param.rgb_intrinsic.cx
    intrinsic[1, 2] = camera_param.rgb_intrinsic.cy

    # get Homogenous coordinates of 2D point camera
    x, y = xy
    p_c_H = np.array([x, y, 1.]).reshape(3, 1) # shape: (3, N), N=1

    # so 3D point in the camera coordinate frame is given by:
    # tensor shape changes: 3x3 matmul 3xN -> 3xN
    p_c_w = np.matmul(np.linalg.inv(intrinsic), p_c_H) * depth

    # now you can do np.dot(camera_Rt, p_c_w_H) to get 3D point in the world coordinate frame

    # See also an example for CARLA is here: https://github.com/JunweiLiang/Multiverse/blob/master/forking_paths_dataset/code/utils.py#L205

    return p_c_w.squeeze().tolist()




if __name__ == "__main__":
    args = parser.parse_args()

    # Assume we are connected to a depth camera
    # lazy import API for specified depth camera
    if args.camera_type == "orbbec":

        # install pyorbbecs through here:
        # https://github.com/orbbec/pyorbbecsdk
        # need to build a local wheel

        from pyorbbecsdk import Pipeline
        from pyorbbecsdk import Config
        from pyorbbecsdk import OBSensorType
        from pyorbbecsdk import OBAlignMode
        from pyorbbecsdk import OBFormat

        # example from https://github.com/orbbec/pyorbbecsdk/blob/main/examples/depth_color_sync_align_viewer.py
        try:
            pipeline = Pipeline()
            config = Config()

            # 1920x1080 only supports 15 fps for Femolt Bolt
            color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(1280, 960, OBFormat.RGB, 30)
            depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()



            # color profile : 1920x1080@15_OBFormat.RGB
            # default color profile : 1280x960@30_OBFormat.MJPG
            print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                   color_profile.get_height(),
                                                   color_profile.get_fps(),
                                                   color_profile.get_format()))
            # depth profile : 640x576@15_OBFormat.Y16
            print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                   depth_profile.get_height(),
                                                   depth_profile.get_fps(),
                                                   depth_profile.get_format()))


            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            # HW_MODE does not work for Femolt Bolt
            config.set_align_mode(OBAlignMode.SW_MODE) # align depth to the color image, at 15 fps

            pipeline.enable_frame_sync()
            pipeline.start(config)

            camera_param = pipeline.get_camera_param()

            # 坐标系原点设置： https://www.orbbec.com/documentation-mega/coordinate-systems/
            # camera param 获取: https://github.com/orbbec/pyorbbecsdk/blob/main/test/test_pipeline.py#L35
            """
            print(camera_param.depth_intrinsic)
            print(camera_param.rgb_intrinsic)
            print(camera_param.depth_distortion)
            print(camera_param.rgb_distortion)
            print(camera_param.transform)

            print(camera_param.rgb_intrinsic.fx)

            <OBCameraIntrinsic fx=997.648743 fy=996.949890 cx=632.307373 cy=490.477325 width=1280 height=960>
            <OBCameraIntrinsic fx=997.648743 fy=996.949890 cx=632.307373 cy=490.477325 width=1280 height=960>
            <OBCameraDistortion k1=0.073824 k2=-0.100994 k3=0.040822 k4=0.000000 k5=0.000000 k6=0.000000 p1=-0.000142 p2=-0.000074>
            <OBCameraDistortion k1=0.073824 k2=-0.100994 k3=0.040822 k4=0.000000 k5=0.000000 k6=0.000000 p1=-0.000142 p2=-0.000074>
            <OBD2CTransform rot=[1, 0, 0, 0, 1, 0, 0, 0, 1]
            transform=[0, 0, 0]
            997.6487426757812
            """

        except Exception as e:
            print(e)
            sys.exit()

    else:
        print("not supported camera.")
        sys.exit()

    print("Now showing the camera stream. press Q to exit.")
    start_time = time.time()
    frame_count = 0
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames(100)  # maximum delay in milliseconds
            if frames is None:
                continue

            # unlike realsense, the frames should be aligned by now
            aligned_frames = frames

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_data = get_orbbec_depth_data(depth_frame)
            color_data = get_orbbec_color_data(color_frame)
            #print(depth_data[depth_data != 0]) # in 米
            #print(depth_data.shape) # (960, 1280)
            #print(color_data.shape) # (960, 1280, 3)

            # showing two points' depth
            point1 = (400, 400)  # (y, x)
            point2 = (480, 640)

            # depth in mm
            color_image, depth1 = show_point_depth(point1, depth_data, color_data)
            color_image, depth2 = show_point_depth(point2, depth_data, color_data)

            point1_3d = deproject_pixel_to_point(camera_param, (point1[1], point1[0]), depth1)
            point2_3d = deproject_pixel_to_point(camera_param, (point2[1], point2[0]), depth2)
            point1_3d_m = deproject_pixel_to_point_matmul(camera_param, (point1[1], point1[0]), depth1)
            print(point1_3d, point1_3d_m)
            np.testing.assert_allclose(point1_3d, point1_3d_m)

            # 计算这两点的实际距离
            #print(point1_3d, point2_3d)
            dist_between_point1_point2 = np.linalg.norm(np.array(point1_3d) - np.array(point2_3d))

            mid_point_xy = ( int((point2[1] + point1[1])/2.), int((point2[0] + point1[0])/2.) + 100)
            color_image = cv2.putText(
                color_image, "dist 1to2: %.2f meters" % dist_between_point1_point2,
                mid_point_xy, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=(0, 0, 255), thickness=2)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data*1000., alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            image = np.hstack((color_image, depth_colormap))
            image = image_resize(image, width=1280, height=None)

            # show the fps
            current_time = time.time()
            frame_count += 1
            fps = frame_count / (current_time - start_time)
            image = cv2.putText(
                image, "FPS: %d" % int(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)

            # Show the image
            cv2.imshow('RGB and Depth Stream', image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
