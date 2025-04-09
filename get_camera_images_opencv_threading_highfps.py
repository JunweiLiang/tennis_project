
# coding=utf-8
# simple camera image grabbing exercise, from laptop cam, web cam

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse
import time
import threading
import queue
import datetime

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
            # 200 fps for 100 seconds
            self.frame_queue = queue.Queue(maxsize=20000)  # Can tune based on memory & burst duration

            fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID is faster than MJPG
            #fourcc = cv2.VideoWriter_fourcc(*'MP4V') # speed seems the same as above and file size similar
            #fourcc = cv2.VideoWriter_fourcc(*'H264') # this you will need to install openh264 for ffmpeg
            self.writer = cv2.VideoWriter(output, fourcc, video_fps, (w, h))
            self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
            self.writer_thread.start()

            self.monitor_thread = threading.Thread(target=self._monitor_queue, daemon=True)
            self.monitor_thread.start()

        self.read_thread = threading.Thread(target=self.update, daemon=True)
        self.read_thread.start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()
            self.frame_count += 1 # count the actual frame we get from opencv
            if self.save_video and self.ret:
                try:
                    # put a timestamp for the frame for possible synchronization
                    # and a frame index to look up depth data
                    date_time = str(datetime.datetime.now())
                    self.frame_queue.put_nowait((self.frame.copy(), date_time, self.frame_count))  # Don't block the capture loop
                except queue.Full:
                    # Drop frames if queue is full
                    pass

    def _write_loop(self):
        # after stop, this will continue to write until all frames are written
        while not self.stopped or not self.frame_queue.empty():
            try:
                frame, date_time, frame_index = self.frame_queue.get(timeout=0.1)
                frame = cv2.putText(
                    frame, "#%d: %s" % (frame_index, date_time),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
                self.writer.write(frame)
            except queue.Empty:
                continue

    def _monitor_queue(self):
        while not self.stopped:
            if self.save_video:
                usage = self.frame_queue.qsize()
                percent = 100.0 * usage / self.frame_queue.maxsize
                # this will be printed in the same line
                print(f"\r[Frame queue usage] {usage:5d}/{self.frame_queue.maxsize} ({percent:5.1f}%)", end="")
            time.sleep(0.5)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.read_thread.join()
        self.stream.release()
        if self.save_video:
            print("\n[INFO] Waiting for video writer to flush remaining frames...")
            self.writer_thread.join()
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






