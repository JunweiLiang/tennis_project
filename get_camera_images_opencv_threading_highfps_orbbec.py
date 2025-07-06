# coding=utf-8
# Adapt code for PyOrbbec depth cameras to read RGB frames
# Initializes the pipeline without specifying device_uri, using the first available device.

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

# Removed --device_uri argument
parser.add_argument("--fps", type=int, default=60,
                    help="FPS for the RGB stream. Orbbec cameras have specific supported FPS values.")
parser.add_argument("--h", type=int, default=800,
                    help="Image height for RGB stream (e.g., 720 for 1280x720, 1080 for 1920x1080).")
parser.add_argument("--w", type=int, default=1280,
                    help="Image width for RGB stream (e.g., 1280 for 1280x720, 1920 for 1920x1080).")
parser.add_argument("--no_show_video", action="store_true")
parser.add_argument("--save_video", action="store_true")

parser.add_argument("--write_video_fps", default=30.0, type=float)
parser.add_argument("--write_video_path", default="output_orbbec_rgb.avi")

class OrbbecCameraStream:
    # Removed device_uri from __init__ arguments
    def __init__(self, fps=30, h=800, w=1280, save_video=False, video_fps=30.0, output="output_orbbec_rgb.avi"):
        self.context = ob.Context()
        self.pipeline = None
        self.config = None

        try:
            # Initialize pipeline without specifying a device_uri.
            # This will automatically open the first available Orbbec device.
            self.pipeline = ob.Pipeline()
            print("Orbbec pipeline initialized, attempting to open the first available device.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Orbbec pipeline: {e}")

        self.config = ob.Config()

        # Enable the color stream
        # Orbbec Gemini 336L, global shutter, RGB max at 1280x800 @60fps, depth max at 1280x800 @30fps
        fps = int(fps)
        w = int(w)
        h = int(h)
        color_profile = self.pipeline.get_stream_profile_list(
            ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
                w, h, ob.OBFormat.MJPG, fps)

        #depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(1280, 800, OBFormat.Y16, 30)
        print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                               color_profile.get_height(),
                                               color_profile.get_fps(),
                                               color_profile.get_format()))

        self.config.enable_stream(color_profile)
        self.stream_format = color_profile.get_format() # Store the format

        self.frame = None
        self.stopped = False
        self.frame_count = 0
        self.fps_actual = color_profile.get_fps() # Store the actual FPS
        self.width_actual = color_profile.get_width()
        self.height_actual = color_profile.get_height()

        # you can enable depth stream and sync them here

        # Start the pipeline
        self.pipeline.start(self.config)

        self.read_thread = threading.Thread(target=self.update, daemon=True)
        self.read_thread.start()

        # intrinsics
        self.camera_param = self.pipeline.get_camera_param()

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

    def update(self):
        while not self.stopped:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames(100)  # maximum delay in milliseconds
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

                if self.stream_format == ob.OBFormat.MJPG:
                    # Decode MJPG data using OpenCV
                    color_image = cv2.imdecode(color_image_data, cv2.IMREAD_COLOR)
                    if color_image is None:
                        print("Failed to decode MJPG frame.")
                        continue
                    # imdecode already returns BGR, so no need for cvtColor if it's already BGR
                    color_image_bgr = color_image
                elif self.stream_format == ob.OBFormat.RGB:
                    # For RGB888, reshape and then convert to BGR
                    color_image = color_image_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
                    color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                else:
                    # Handle other formats if necessary, or print a warning
                    print(f"Unsupported color format: {self.stream_format}")
                    continue

                self.frame = color_image_bgr
                self.frame_count += 1
                if self.save_video:
                    try:
                        date_time = str(datetime.datetime.now())
                        self.frame_queue.put_nowait((self.frame.copy(), date_time, self.frame_count))
                    except queue.Full:
                        pass # Drop frames if queue is full

            time.sleep(0.001) # Small sleep to prevent busy-waiting


    def read(self):
        # Return the latest frame received by the callback
        return self.frame

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
        # Pass arguments directly to the constructor, no device_uri needed
        stream = OrbbecCameraStream(
            fps=args.fps, h=args.h, w=args.w,
            save_video=args.save_video, video_fps=args.write_video_fps, output=args.write_video_path
        )
        print(f"Successfully initialized Orbbec RGB stream: {stream.width_actual}x{stream.height_actual} @ {stream.fps_actual} FPS")

        print("Now showing the camera stream. Press Q to exit.")

        start_time = time.time()
        display_frame_count = 0

        while True:

            if args.no_show_video:
                time.sleep(0.001) # Small sleep to prevent busy-waiting
                continue

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
    finally:
        if stream:
            stream.stop()
        cv2.destroyAllWindows()
