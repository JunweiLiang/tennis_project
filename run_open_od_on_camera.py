# coding=utf-8
# simple camera image grabbing exercise, from laptop cam, web cam
# and run object detection on the images and visualize

# you can install cv2 with $ pip install opencv-python
import cv2
import argparse

# This is a good open-source wrapper
# pip install ultralytics
# see tutorial here
#   1. https://docs.ultralytics.com/models/yolov9/
#   2. https://medium.com/@Mert.A/how-to-use-yolov9-for-object-detection-93598ad88d7d
#   3. here for YOLO-world: https://docs.ultralytics.com/models/yolo-world/#set-prompts
from ultralytics import YOLO

parser = argparse.ArgumentParser()

parser.add_argument("--cam_num", type=int, default=0,
        help="camera num")
parser.add_argument("--model_name", default="yolov8s-worldv2.pt")
parser.add_argument("--prompts", action="append", help="set prompt with quotes", default=None)
parser.add_argument("--conf", default=0.2, type=float, help="confident score threshold")
# example
# tennis_project$ python run_open_od_on_camera.py --prompts "blue box" --prompts "yellow box"

parser.add_argument("--output_image", help="grab a image from camera and save to this file")

# example run on a macbook
# junweiliang@work_laptop:~/Desktop/projects/tennis_project$ python run_od_on_camera.py  --cam_num 1
# if you see
#   [ WARN:0@0.012] global cap_v4l.cpp:999 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
# 可能需要在笔记本电脑上登录一下你的账号，唤醒一下摄像头


def run_od_on_image(
        frame_cv2, od_model,
        classes=None, conf=0.3,
        bbox_thickness=4, text_thickness=2, font_size=2):
    """
        Run object detection inference and visualize in the image.
        Return: the visualization image and the detection results
    """

    # see here for inference arguments
    # https://docs.ultralytics.com/modes/predict/#inference-arguments
    results = od_model.predict(
        frame_cv2,
        classes=classes,  # you can specify the classes you want
        # see here for coco class indexes [0-79], 0 is person: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
        #classes=[0, 32], # detect person and sports ball only
        conf=conf)

    # see here for the API documentation of results
    # https://docs.ultralytics.com/modes/predict/#working-with-results
    result = results[0] # we run it on single image
    for box in result.boxes:
        bbox = [int(x) for x in box.xyxy[0]]
        bbox_color = (0, 255, 0) # BGR
        frame_cv2 = cv2.rectangle(
                frame_cv2,
                tuple(bbox[0:2]), tuple(bbox[2:4]),
                bbox_color, bbox_thickness)

        frame_cv2 = cv2.putText(
                frame_cv2, "%s" % result.names[int(box.cls[0])],
                (bbox[0], bbox[1] - 10),  # specify the bottom left corner
                cv2.FONT_HERSHEY_PLAIN, font_size,
                bbox_color, text_thickness)
    return frame_cv2, results



if __name__ == "__main__":
    args = parser.parse_args()

    # initialize the object detection model
    # this will auto download the YOLOv9 checkpoint
    # see here for all the available models: https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset
    model = YOLO(args.model_name)

    # set the open vocabulary prompt classes (a list)
    # if not set, the default prompt are the 80 COCO classes
    #model.set_classes(["blue box"]) # this may start to download CLIP stuff
    if args.prompts is not None:
        model.set_classes(args.prompts)

    # 1. assume we are in a laptop, grab an image from the web camera

    ## note that if you are on Macbook, and you have a iphone, camera 0 might be your iphone camera!!
    cam_num = args.cam_num

    cam = cv2.VideoCapture(cam_num)

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


            print("Now showing the camera stream. press Q to exit.")

            while True:
                ret, frame = cam.read()
                if not ret:
                    raise Exception("Error: Could not read frame from webcam.")

                frame, det_results = run_od_on_image(frame, model, conf=args.conf)

                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.output_image:
                cv2.imwrite(args.output_image, frame)
                print("saved mediapiped image from web cam to %s" % args.output_image)

    finally:
        # release window
        cam.release()
        cv2.destroyAllWindows()
