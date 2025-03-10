# coding=utf-8
# simple camera image grabbing exercise, from laptop cam, web cam

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--show_streaming",
        action="store_true",
        help="this will also open a window to see the camera stream, q to exit")
parser.add_argument("--cam_num", type=int, default=0,
        help="camera num")
parser.add_argument("output_image", help="grab a image from camera and save to this file")

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

    cam = cv2.VideoCapture(cam_num)

    if cam is None or not cam.isOpened():
        print("failed to grab camera %s" % cam_num)

    else:
        # you can print out the info about your camera
        print("------ info about your camera ----")
        print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cam.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("CAP_PROP_FPS : '{}'".format(cam.get(cv2.CAP_PROP_FPS)))
        print("CAP_PROP_POS_MSEC : '{}'".format(cam.get(cv2.CAP_PROP_POS_MSEC)))
        print("CAP_PROP_FRAME_COUNT  : '{}'".format(cam.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("CAP_PROP_BRIGHTNESS : '{}'".format(cam.get(cv2.CAP_PROP_BRIGHTNESS)))
        print("CAP_PROP_CONTRAST : '{}'".format(cam.get(cv2.CAP_PROP_CONTRAST)))
        print("CAP_PROP_SATURATION : '{}'".format(cam.get(cv2.CAP_PROP_SATURATION)))
        print("CAP_PROP_HUE : '{}'".format(cam.get(cv2.CAP_PROP_HUE)))
        print("CAP_PROP_GAIN  : '{}'".format(cam.get(cv2.CAP_PROP_GAIN)))
        print("CAP_PROP_CONVERT_RGB : '{}'".format(cam.get(cv2.CAP_PROP_CONVERT_RGB)))
        print("------- end camera info ----")


        if args.show_streaming:

            print("Now showing the camera stream. press Q to exit.")

            while True:
                ret, frame = cam.read()
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    sys.exit()

                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        result, image = cam.read()

        if result:
            cv2.imwrite(args.output_image, image)
            print("saved image from web cam to %s" % args.output_image)
        else:
            print("Failed to grab image from cam %s" % cam_num)

        # release window
        cam.release()
        cv2.destroyAllWindows()
