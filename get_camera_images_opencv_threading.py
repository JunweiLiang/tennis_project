
# coding=utf-8
# simple camera image grabbing exercise, from laptop cam, web cam

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import time
import threading

parser = argparse.ArgumentParser()

parser.add_argument("--cam_num", type=int, default=0,
        help="camera num")

class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

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

    stream = WebcamStream(cam_num)

    print("Now showing the camera stream. press Q to exit.")
    # Use the threaded webcam reader
    start_time = time.time()
    frame_count = 0
    while True:
        frame = stream.read()
        frame_count += 1
        current_time = time.time()
        fps = int(frame_count / (current_time - start_time))
        frame = cv2.putText(
            frame, "FPS: %d" % int(fps),
            (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(0, 0, 255), thickness=2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # release window
    stream.stop()
    cv2.destroyAllWindows()






