# ******************************************************************************
# modified by Teli Ma
# ******************************************************************************
import argparse
import sys

import cv2
import numpy as np
from typing import Union, Any, Optional


from pyorbbecsdk import *
from pyorbbecsdk import FormatConvertFilter, VideoFrame
from pyorbbecsdk import OBFormat, OBConvertFormat
# from utils import frame_to_bgr_image
import pdb
import os

ESC_KEY = 27

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

def main(argv):
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode",
                        help="align mode, HW=hardware mode,SW=software mode,NONE=disable align",
                        type=str, default='HW')
    parser.add_argument("-s", "--enable_sync", help="enable sync", type=bool, default=True)
    args = parser.parse_args()
    align_mode = args.mode
    enable_sync = args.enable_sync
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1280, 960, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        #color profile : 1280x960@30_OBFormat.RGB
        #depth profile : 640x576@15_OBFormat.Y16
        print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                   color_profile.get_height(),
                                                   color_profile.get_fps(),
                                                   color_profile.get_format()))
        print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                   depth_profile.get_height(),
                                                   depth_profile.get_fps(),
                                                   depth_profile.get_format()))
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return

    if align_mode == 'HW':
        if device_pid == 0x066B:
            # Femto Mega does not support hardware D2C, and it is changed to software D2C
            config.set_align_mode(OBAlignMode.SW_MODE)
        else:
            config.set_align_mode(OBAlignMode.HW_MODE)
    elif align_mode == 'SW':
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.DISABLE)
    if enable_sync:
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(e)
    try:
        pipeline.start(config)
    except Exception as e:
        print(e)
        return
    save_path = os.path.join('vis_input', 'test0')
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            # pdb.set_trace()
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            # pdb.set_trace()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data * scale
            depth_data = np.uint16(depth_data)
            print(depth_data.shape, color_image.shape)
            break
            depth_image = cv2.cvtColor(depth_data, cv2.COLOR_BAYER_BG2GRAY)

            # pdb.set_trace()
            # depth_data = depth_data.astype(np.float32) * scale
            
            # depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            # overlay color image on depth image
            # depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
            cv2.imshow('color ', color_image)
            cv2.imshow("depth ", depth_image)
            print(color_image.shape, depth_image.shape)  # same HW
            # cv2.imshow('depth ', depth_data)
            #cv2.imwrite(os.path.join(save_path, 'color.png'), color_image)
            #cv2.imwrite(os.path.join(save_path, 'depth.png'), depth_image)
            # break
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
