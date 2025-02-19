# coding=utf-8
# simple camera image grabbing exercise, from laptop cam, web cam

# you can install cv2 with $ pip install opencv-python
import av
import cv2

import sys
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("--show_streaming",
        action="store_true",
        help="this will also open a window to see the camera stream, q to exit")
parser.add_argument("--cam_num", type=int, default=0,
        help="camera num")
parser.add_argument("--output_image", default="", help="grab a image from camera and save to this file")

# 1. example use to get an image from the laptop camera
# (base) junweil@precognition-laptop2:~$ python ~/projects/tennis_project/get_camera_image.py Downloads/output.png
# if you see
#   [ WARN:0@0.012] global cap_v4l.cpp:999 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
# 可能需要在笔记本电脑上登录一下你的账号，唤醒一下摄像头

if __name__ == "__main__":
    args = parser.parse_args()

    # 1. assume we are in a laptop, grab an image from the web camera

    ## note that if you are on Macbook, and you have a iphone, camera 0 might be your iphone camera!!
    cam_num = args.cam_num

    # use v4l2-ctl --list-formats-ext to see available format from webcamera
    """
        DJI Action4 webcam mode

            (base) junweil@home-lab:~/projects/tennis_project$ v4l2-ctl --list-formats-ext
                ioctl: VIDIOC_ENUM_FMT
                    Type: Video Capture

                    [0]: 'MJPG' (Motion-JPEG, compressed)
                        Size: Discrete 1280x720
                            Interval: Discrete 0.033s (30.000 fps)
                        Size: Discrete 1920x1080
                            Interval: Discrete 0.033s (30.000 fps)
                    [1]: 'H264' (H.264, compressed)
                        Size: Discrete 1280x720
                            Interval: Discrete 0.033s (30.000 fps)
                        Size: Discrete 1920x1080
                            Interval: Discrete 0.033s (30.000 fps)

        500 RMB camera:
            (base) junweil@home-lab:~/projects/tennis_project$ v4l2-ctl --list-formats-ext
            ioctl: VIDIOC_ENUM_FMT
                Type: Video Capture

                [0]: 'MJPG' (Motion-JPEG, compressed)
                    Size: Discrete 3840x3040
                        Interval: Discrete 0.050s (20.000 fps)
                    Size: Discrete 3840x2160
                        Interval: Discrete 0.033s (30.000 fps)
                        Interval: Discrete 0.050s (20.000 fps)
                    ...
                    Size: Discrete 1920x1080
                        Interval: Discrete 0.008s (120.000 fps)
                        Interval: Discrete 0.017s (60.000 fps)
                    Size: Discrete 1280x960
                        Interval: Discrete 0.008s (120.000 fps)
                    Size: Discrete 1280x720
                        Interval: Discrete 0.008s (120.000 fps)
                [1]: 'YUYV' (YUYV 4:2:2)
                    Size: Discrete 3840x3040
                        Interval: Discrete 1.000s (1.000 fps)
                    Size: Discrete 3840x2160
                        Interval: Discrete 1.000s (1.000 fps)
                    Size: Discrete 1920x1080
                        Interval: Discrete 0.200s (5.000 fps)

        $ conda install av -c conda-forge
        # use this to install pyav with an independant ffmpeg, otherwise opencv imshow got stuck

    """
    cam = av.open("/dev/video%s" % cam_num, format="v4l2", options={
        "video_size": "1920x1080",  # Set resolution
        "framerate": "120",         # Set FPS
        "input_format": "mjpeg",  # Force MJPEG
    })

    if cam is None:
        print("failed to grab camera %s" % cam_num)

    else:
        # Get the first video stream
        video_stream = cam.streams.video[0]

        # Print actual FPS and resolution (to verify settings)
        print(f"Actual FPS: {video_stream.average_rate} FPS")
        print(f"Resolution: {video_stream.width}x{video_stream.height}")


        if args.show_streaming:

            print("Now showing the camera stream. press Q to exit.")
            start_time = time.time()
            frame_count = 0
            for frame in cam.decode(video=0):
                frame = frame.to_ndarray(format="bgr24")  # Convert PyAV frame to OpenCV format
                frame_count += 1
                current_time = time.time()
                fps = int(frame_count / (current_time - start_time))

                frame = cv2.putText(
                    frame, "FPS: %d" % int(fps),
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)

                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:

            image = cam.decode(video=0)

            if image:
                if args.output_image != "":
                    cv2.imwrite(args.output_image, image)
                    print("saved image from web cam to %s" % args.output_image)
            else:
                print("Failed to grab image from cam %s" % cam_num)

        # release window
        cam.close()
        cv2.destroyAllWindows()
