# coding=utf-8

import cv2
import time
import numpy as np

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

printed = False
def print_once(string):
    global printed
    if not printed:
        print(string)
        printed = True


def run_od_on_image(
        frame_cv2, od_model,
        classes=[], conf=0.5,
        bbox_thickness=4, text_thickness=2, font_size=2):
    """
        run object detection inference and visualize in the image
    """

    # see here for inference arguments
    # https://docs.ultralytics.com/modes/predict/#inference-arguments
    results = od_model.predict(
        frame_cv2,
        classes=None if len(classes)==0 else classes,  # you can specify the classes you want
        # see here for coco class indexes [0-79], 0 is person: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
        #classes=[0, 32], # detect person and sports ball only
        conf=conf,
        #half=True
        )

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

def run_od_track_on_image(
        frame_cv2, od_model, track_history,
        classes=[], conf=0.5,
        tracker_yaml="bytetrack.yaml", # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml
        bbox_thickness=4, text_thickness=2, font_size=2):
    """
        run object detection and tracking on a new frame, and visualize
    """

    # see here for inference arguments
    # https://docs.ultralytics.com/modes/track/#tracking
    results = od_model.track(
        frame_cv2,
        tracker=tracker_yaml,
        #tracker="botsort.yaml",
        classes=None if len(classes)==0 else classes,  # you can specify the classes you want
        # see here for coco class indexes [0-79], 0 is person: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
        #classes=[0, 32], # detect person and sports ball only
        conf=conf,
        iou=0.5, #NMS. Lower values result in fewer detections by eliminating overlapping boxes
        persist=True
        )

    # see here for the API documentation of results
    # https://docs.ultralytics.com/modes/predict/#working-with-results
    result = results[0]

    time_now = time.time()

    # Get the boxes and track IDs for ploting the lines
    boxes = result.boxes.xywh.cpu()
    # only visualize when there are tracks
    if result.boxes.id is not None:
        boxes_xyxy = result.boxes.xyxy.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, box_xyxy, track_id, cls_id in zip(boxes, boxes_xyxy, track_ids, classes):
            center_x, center_y, w, h = box
            x1, y1, x2, y2 = box_xyxy

            bbox_color = (255, 0, 0) # BGR

            frame_cv2 = cv2.rectangle(
                    frame_cv2,
                    (int(x1), int(y1)), (int(x2), int(y2)),
                    bbox_color, bbox_thickness)

            frame_cv2 = cv2.putText(
                    frame_cv2, "%s #%d" % (result.names[cls_id], track_id),
                    (int(x1), int(y1) - 10),  # specify the bottom left corner
                    cv2.FONT_HERSHEY_PLAIN, font_size,
                    bbox_color, text_thickness)

            # plot the trace
            track = track_history[track_id]
            track.append((int(center_x), int(center_y), cls_id, time_now))
            if len(track) > 30:
                track.pop(0)

            points = np.hstack([ (x[0], x[1]) for x in track]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_cv2, [points], isClosed=False, color=(230, 230, 230), thickness=8)

    return frame_cv2, results

