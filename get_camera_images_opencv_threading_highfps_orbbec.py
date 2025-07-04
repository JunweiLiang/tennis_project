
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

import numpy as np

# Import pyorbbecsdk
import pyorbbecsdk as ob

parser = argparse.ArgumentParser()

parser.add_argument("--device_uri", type=str, default=None,
                    help="Orbbec camera device URI (e.g., 'usb://bus-port'). If None, opens the first available device.")

# for Orbbec Gemini 336L, global shutter, RGB max at 1280x800 @60fps, depth max at 1280x800 @30fps
parser.add_argument("--fps", type=int, default=60,
        help="fps. ")
parser.add_argument("--h", type=int, default=800,
        help="image height ")
parser.add_argument("--w", type=int, default=1280,
        help="image width ")
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--write_video_fps", default=30.0, type=float)
parser.add_argument("--write_video_path", default="output.avi")

class OrbbecCameraStream:
    def __init__(self, device_uri=None, fps=60, h=720, w=1280, save_video=False, video_fps=30.0, output="output_orbbec_rgb.avi"):
        self.context = ob.Context()
        self.device = None
        self.pipeline = None
        self.config = None

        if device_uri:
            try:
                self.device = self.context.open_device_by_uri(device_uri)
                print(f"Opened Orbbec device with URI: {device_uri}")
            except Exception as e:
                print(f"Error opening device with URI {device_uri}: {e}. Attempting to open the first available device.")
                self._open_first_device()
        else:
            self._open_first_device()

        if not self.device:
            raise RuntimeError("No Orbbec device found or could be opened.")

        self.pipeline = ob.Pipeline(self.device)
        self.config = ob.Config()

        # Enable the color stream
        color_profiles = self.pipeline.get_stream_profiles(ob.StreamType.COLOR)
        selected_color_profile = None
        for profile in color_profiles:
            if isinstance(profile, ob.VideoStreamProfile):
                # Try to find a matching profile
                if profile.width() == w and profile.height() == h and profile.fps() == fps and profile.format() == ob.Format.RGB888:
                    selected_color_profile = profile
                    break

        if not selected_color_profile:
            # If exact match not found, try to find a suitable one and warn
            print(f"Warning: Exact RGB profile (W:{w}, H:{h}, FPS:{fps}, Format:RGB888) not found. Trying to find a close match.")
            for profile in color_profiles:
                if isinstance(profile, ob.VideoStreamProfile) and profile.format() == ob.Format.RGB888:
                    selected_color_profile = profile
                    print(f"Selected RGB profile: W:{profile.width()}, H:{profile.height()}, FPS:{profile.fps()}, Format:{profile.format()}")
                    break

        if not selected_color_profile:
            raise RuntimeError("No suitable RGB888 stream profile found on the device.")

        self.config.enable_stream(selected_color_profile)

        self.frame = None
        self.stopped = False
        self.frame_count = 0
        self.fps_actual = selected_color_profile.fps() # Store the actual FPS
        self.width_actual = selected_color_profile.width()
        self.height_actual = selected_color_profile.height()

        self.save_video = save_video
        self.writer = None
        if save_video:
            self.frame_queue = queue.Queue(maxsize=20000)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(output, fourcc, video_fps, (self.width_actual, self.height_actual))
            self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
            self.writer_thread.start()
            self.monitor_thread = threading.Thread(target=self._monitor_queue, daemon=True)
            self.monitor_thread.start()

        self.read_thread = threading.Thread(target=self.update, daemon=True)
        self.read_thread.start()

        # Start the pipeline with the callback
        self.pipeline.start(self._frame_callback, self.config)

    def _open_first_device(self):
        device_list = self.context.query_device_list()
        if device_list:
            self.device = device_list.get_device_by_index(0)
            print(f"Opened the first available Orbbec device: {self.device.get_usb_info().name}")
        else:
            print("No Orbbec devices found.")

    def _frame_callback(self, frame_set):
        color_frame = frame_set.get_color_frame()
        if color_frame:
            color_image_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            # Reshape to BGR for OpenCV (Orbbec RGB888 is RGB, OpenCV expects BGR)
            color_image = color_image_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            self.frame = color_image_bgr
            self.frame_count += 1
            if self.save_video:
                try:
                    date_time = str(datetime.datetime.now())
                    self.frame_queue.put_nowait((self.frame.copy(), date_time, self.frame_count))
                except queue.Full:
                    pass # Drop frames if queue is full

    def update(self):
        # This thread now primarily manages the 'stopped' state and can be used for other background tasks
        # The actual frame grabbing is handled by the Orbbec SDK's internal thread and our callback
        while not self.stopped:
            time.sleep(0.001) # Small sleep to prevent busy-waiting

    def _write_loop(self):
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
                print(f"\r[Frame queue usage] {usage:5d}/{self.frame_queue.maxsize} ({percent:5.1f}%)", end="")
            time.sleep(0.5)

    def read(self):
        # Return the latest frame received by the callback
        return self.frame

    def stop(self):
        self.stopped = True
        if self.pipeline:
            self.pipeline.stop()

        # Ensure read_thread is joined
        self.read_thread.join()

        if self.save_video:
            print("\n[INFO] Waiting for video writer to flush remaining frames...")
            self.writer_thread.join()
            self.writer.release()

        # Orbbec context and device are managed internally by the pipeline's stop

if __name__ == "__main__":
    args = parser.parse_args()

    if args.save_video:
        print("write video @%d fps to %s" % (args.write_video_fps, args.write_video_path))

    stream = None
    try:
        stream = OrbbecCameraStream(
            device_uri=args.device_uri,
            fps=args.fps, h=args.h, w=args.w,
            save_video=args.save_video, video_fps=args.write_video_fps, output=args.write_video_path
        )
        print(f"Successfully initialized Orbbec RGB stream: {stream.width_actual}x{stream.height_actual} @ {stream.fps_actual} FPS")

        print("Now showing the camera stream. Press Q to exit.")

        start_time = time.time()
        display_frame_count = 0

        while True:
            frame = stream.read()

            if frame is not None:
                display_frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > 0:
                    display_fps = int(display_frame_count / elapsed_time)
                else:
                    display_fps = 0 # Avoid division by zero at the very beginning

                frame = cv2.putText(
                    frame, "Display FPS: %d (Actual Stream FPS: %d)" % (display_fps, stream.fps_actual),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
                frame = cv2.putText(
                    frame, "Total Frames Read: %d" % stream.frame_count,
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)

                cv2.imshow("Orbbec RGB Stream", frame)
            else:
                print("Waiting for first frame from Orbbec camera...")
                time.sleep(0.1) # Wait a bit if no frame is available yet

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




