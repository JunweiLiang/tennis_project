# coding=utf-8
# simple camera image grabbing exercise, from depth camera like Orbbec Femto Bolt
# https://github.com/orbbec/pyorbbecsdk/blob/main/docs/README_EN.md

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import numpy as np

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
        (point[1], point[0]), radius=2, color=(0, 255, 0), thickness=2)
    color_image = cv2.putText(
        color_image, "depth: %smm" % depth,
        (point[1], point[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, color=(0, 255, 0), thickness=2)
    return color_image, depth

def get_orbbec_depth_data(orbbec_depth_frame):
    width = orbbec_depth_frame.get_width()
    height = orbbec_depth_frame.get_height()
    scale = orbbec_depth_frame.get_depth_scale()

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    depth_data = depth_data.astype(np.float32) * scale

    return depth_data

def get_point_position(data_dir, xy):
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    # pdb.set_trace()
    # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    fx, fy = 999.273682, 998.180237
    cx, cy = 638.187988, 477.954865
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    # xmin, xmax = -0.19, 0.12
    # ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    #mask = (points_z >= 0) &amp; (points_z &lt; 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points_origin = points
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    center = points_origin[xy[0], xy[1]]
    print('point:', center)

    return center

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
            config.set_align_mode(OBAlignMode.SW_MODE)
            pipeline.enable_frame_sync()
            pipeline.start(config)

        except Exception as e:
            print(e)
            sys.exit()

    else:
        print("not supported camera.")
        sys.exit()


    print("Now showing the camera stream. press Q to exit.")
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
            #print(depth_data[depth_data != 0]) # in 毫米
            print(depth_data.shape) # (576, 640)
            print(color_data.shape) # (1080, 1920, 3)

            break
            # junwei: the color_intrin and depth_intrin are the same as they are aligned.
            #### 获取相机参数 ####
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
            aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            # [ 640x480  p[325.217 238.38]  f[385.38 384.848]  Inverse Brown Conrady [-0.0565123 0.067672 0.000208852 0.000719325 -0.0218305] ]
            color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

            #print(depth_intrin, color_intrin)


            # (480, 640, 3), (480, 640)
            # depth_image are in meters
            #print(color_image.shape, depth_image.shape)
            #print(depth_image[240, 320]) # 单位：毫米

            # showing two points' depth
            point1 = (200, 200)  # (y, x)
            point2 = (240, 320)

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

            mid_point_xy = ( int((point2[1] + point1[1])/2.), int((point2[0] + point1[0])/2.) + 50)
            color_image = cv2.putText(
                color_image, "dist 1to2: %.2f meters" % dist_between_point1_point2,
                mid_point_xy, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)

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
