# coding=utf-8
# simple camera image grabbing exercise, from laptop cam, web cam

# you can install cv2 with $ pip install opencv-python
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
parser.add_argument("--set_to_hd_120fps", action="store_true")

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

    #cam = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)
    cam = cv2.VideoCapture(cam_num)

    if cam is None or not cam.isOpened():
        print("failed to grab camera %s" % cam_num)

    else:
        if args.set_to_hd_120fps:
            cam.set(cv2.CV_CAP_PROP_FOURCC, cv2.CV_FOURCC('M', 'J', 'P', 'G'));
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cam.set(cv2.CAP_PROP_FPS, 120)

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

            start_time = time.time()
            frame_count = 0
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    sys.exit()

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

            result, image = cam.read()

            if result:
                if args.output_image != "":
                    cv2.imwrite(args.output_image, image)
                    print("saved image from web cam to %s" % args.output_image)
            else:
                print("Failed to grab image from cam %s" % cam_num)

        # release window
        cam.release()
        cv2.destroyAllWindows()
