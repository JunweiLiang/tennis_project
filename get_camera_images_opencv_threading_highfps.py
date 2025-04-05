
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
parser.add_argument("--fps", type=int, default=60,
        help="fps. ")
parser.add_argument("--h", type=int, default=1080,
        help="image height ")
parser.add_argument("--w", type=int, default=1920,
        help="image width ")
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--write_video_fps", default=30.0, type=float)
parser.add_argument("--write_video_path", default="output.avi")

class WebcamStream:
    def __init__(self, src=0, fps=60, h=1080, w=1920, save_video=False, video_fps=30.0, output="output.avi"):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.frame_count = 0

        self.save_video = save_video
        self.writer = None
        if save_video:
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.writer = cv2.VideoWriter(output, fourcc, video_fps, (w, h))
            if not self.writer:
                print("video writer init failed!")
                sys.exit()

        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()
            self.frame_count += 1 # count the actual frame we get from opencv
            if self.save_video and self.ret and self.writer is not None:
                # this may slow down the FPS
                self.writer.write(self.frame)


    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
        if self.writer:
            self.writer.release()


if __name__ == "__main__":
    args = parser.parse_args()

    # 1. assume we are in a laptop, grab an image from the web camera

    ## note that if you are on Macbook, and you have a iphone, camera 0 might be your iphone camera!!
    cam_num = args.cam_num
    if args.save_video:
        print("write video @%d fps to %s" % (args.write_video_fps, args.write_video_path))


    stream = WebcamStream(cam_num,
        fps=args.fps, h=args.h, w=args.w,
        save_video=args.save_video, video_fps=args.write_video_fps, output=args.write_video_path)

    print("Now showing the camera stream. press Q to exit.")
    # Use the threaded webcam reader
    start_time = time.time()
    while True:
        frame = stream.read()
        frame_count = stream.frame_count
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






