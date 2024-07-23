# coding=utf-8
# simple camera image grabbing exercise, from laptop cam, web cam, and depth camera

# you can install cv2 with $ pip install opencv-python
import cv2

import sys
import argparse

# This is Google's open-source pose estimation package
# $ pip install mediapipe
# This will be running on CPU
import mediapipe as mp

parser = argparse.ArgumentParser()

parser.add_argument("--show_streaming",
        action="store_true",
        help="this will also open a window to see the camera stream, q to exit")
parser.add_argument("--cam_num", type=int, default=0,
        help="camera num")
parser.add_argument("output_image", help="grab a image from camera and save to this file")

# example run on a macbook
# junweiliang@work_laptop:~/Desktop/projects/tennis_project$ python run_pose_est_on_camera.py --show_streaming ~/Downloads/output_pose.png --cam_num 1
# if you see
#   [ WARN:0@0.012] global cap_v4l.cpp:999 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
# 可能需要在笔记本电脑上登录一下你的账号，唤醒一下摄像头


# Initialize MediaPipe Hands.
# will download model if needed
# readme: https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False, # treat it as video stream
    max_num_hands=2,
    model_complexity=0, # 0/1 model,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Initialize MediaPipe Skeleton.
# https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1, # 0/1/2, higher better model
    static_image_mode=False,
    smooth_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
# for drawing the landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def run_mediapipe_on_image(frame_cv2):
    """
        frame_cv2 will be modified in place
    """

    # Process the color image with MediaPipe
    frame_for_mp = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    # Process the image to detect the skeleton and hands
    hand_results = mp_hands.process(frame_for_mp)
    pose_results = pose.process(frame_for_mp)

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_cv2,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style())

    # Draw the pose annotations on the RGB image.
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_cv2,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style())




if __name__ == "__main__":
    args = parser.parse_args()

    # 1. assume we are in a laptop, grab an image from the web camera

    ## note that if you are on Macbook, and you have a iphone, camera 0 might be your iphone camera!!
    cam_num = args.cam_num

    cam = cv2.VideoCapture(cam_num)

    # junwei: use try for more robust code with detailed exception handling
    try:
        if cam is None or not cam.isOpened():
            raise Exception("failed to grab camera %s" % cam_num)

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
                        raise Exception("Error: Could not read frame from webcam.")

                    run_mediapipe_on_image(frame)

                    cv2.imshow("frame", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            result, image = cam.read()

            if result:
                run_mediapipe_on_image(image)
                cv2.imwrite(args.output_image, image)
                print("saved mediapiped image from web cam to %s" % args.output_image)
            else:
                raise Exception("Failed to grab image from cam %s" % cam_num)

    finally:
        # release window
        cam.release()
        cv2.destroyAllWindows()
