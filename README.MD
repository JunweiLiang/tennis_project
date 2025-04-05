# A simple Tennis Tracking/Robot Coach project for undergrads

## Introduction

This repo contains clean exercise code for:

+ grab images from laptop cameras
+ running yolo object detection on web cameras (and count people)
    ```
        $ python run_od_on_camera.py
    ```
+ running open-vocabulary object detection
    ```
        $ python run_open_od_on_camera.py --prompts "blue box" --prompts "yellow box"
    ```
+ running pose estimation on web cameras
+ grab depth images (and compute 3D point distances) using depth cameras (D455i, Femoto Bolt)
+ Show RGB and depth visualization and also save the visualization into a AVI video
    ```
    tennis_project$ python get_depth_images_orbbec.py --save_to_avi orbbec_viz_highres.avi
    tennis_project$ python get_depth_images.py --save_to_avi realsense_viz_highres.avi
    ```
+ Show and save the RGB stream only
    ```
    Realsense D455 1280x720@30fps
    tennis_project$ python get_depth_images.py --save_to_avi realsense_rgb_only.avi --save_data_only
    Femoto Bolt 1280x960@30fps
    tennis_project$ python get_depth_images_orbbec.py --save_to_avi orbbec_rgb_only.avi --save_data_only
    ```

### Pipeline to Do Tennis Ball Speed Estimation with Depth Camera

+ Camera setup: One on a 4-meter tripod tilted down to look like boardcasting angle, only RGB used; One on a 1.5-meter tripod near one corner, we can estimate one player at one side (from baseline to serve line) with a depth camera more reliably.
+ Run depth visualization to make sure the depth camera works using the court lines
    ```
    # baseline to net: 11.88 m
    # baseline to serve line: 5.48m
    # Singles side line to Doubles side line: 1.37m ; this can be used for checking

    # for realsense
    (base) junweil@precognition-laptop2:~/projects/tennis_project$ python get_depth_images.py --save_to_avi speed_rs_depth_vis.avi
    # for orbbec Femoto Bolt
    (base) junweil@precognition-laptop2:~/projects/tennis_project$ python get_depth_images_orbbec.py --save_to_avi speed_orbbec_depth_vis.avi

    ```
+ Run tennis ball speed estimation (visualization and saving to a video file). It is recommended to use detection only, assuming only one ball in sight:
    ```
    # for realsense
    (base) junweil@precognition-laptop2:~/projects/tennis_project$ python run_od_track_speed_on_realsense.py --yolo_model_name yolov10x.pt --det_conf 0.05 --tracker_yaml bytetrack_fastobj.yaml --det_only --save_to_avi speed_rs_det.avi
    # --use_kmh to switch to km/h

    # for orbbec Femoto Bolt
    (base) junweil@precognition-laptop2:~/projects/tennis_project$ python run_od_track_speed_on_orbbec.py --yolo_model_name yolov10x.pt --det_conf 0.05 --tracker_yaml bytetrack_fastobj.yaml --det_only --save_to_avi speed_orbbec_det.avi

    ```

+ The tracking is not good for fast moving object:
    ```
    (base) junweil@precognition-laptop2:~/projects/tennis_project$ python run_od_track_speed_on_realsense.py --yolo_model_name yolov10x.pt --det_conf 0.05 --tracker_yaml bytetrack_fastobj.yaml --save_to_avi speed_rs_track.avi

    # for orbbec Femoto Bolt
    (base) junweil@precognition-laptop2:~/projects/tennis_project$ python run_od_track_speed_on_orbbec.py --yolo_model_name yolov10x.pt --det_conf 0.05 --tracker_yaml bytetrack_fastobj.yaml --save_to_avi speed_orbbec_track.avi
    ```

+ To verify our speed estimation code, we use a 1 meter marker and free drop a ball, which should have a maximum speed of 4.43 m/s. The marker length is 1 meter. See videos in `tennis_est_examples`:
    ```
    (base) junweil@precognition-laptop2:~/projects/tennis/tennis_project$ python run_od_track_speed_on_realsense.py --yolo yolov10m.pt --det_conf 0.01 --det_only --save tennis_est_examples/drop_ball_validate_1meter_4.4ms_yolov10mconf0.01.avi
    ```

### Future Work

+ Use multiple high FPS (>= 60) RGB cameras for speed estimation. The depth camera has low FPS (30) and noisy depths, which makes it unrealiable for ball (fast object) speed estimation.
+ Fine-tune the tennis ball detection/tracking model. Currently the key challenge is to detect fast-moving, small tennis ball.
+ 3D Visualization with multiple cameras. Need to calibrate all the cameras in the scene with their extrinsics, and put the tennis ball into 3D visualization.
+ With the above reliable ball tracking in 3D, we can do ball bounce detection and line calling

### Other potential resources
+ From Roboflow `https://universe.roboflow.com/search?q=class%3Atennis-ball`
+ This repo has a good perception component for tennis `https://core-robotics-lab.github.io/Wheelchair-Tennis-Robot/`

### 04/2025 Adding a script to use high-speed camera
+ We conduct a experiment with two camera and write the video in slow-motion
0. buy this cheap 180fps@1280x1024@global-shutter camera from taobao: `【淘宝】7天无理由退货 https://e.tb.cn/h.6Uvc10ngnzXBaP1?tk=MtJmey8nens MF168 「高帧率180@ fps全局快门摄像头模组1080P高速抓拍USB模块免驱UVC协」`
1. align the two camera like [this](tennis_est_examples/2_highspeed_camera_setup.png)
2. run visualization and saving video to disk (nvme is better):
```
    # the original video frame will run as fast as 180 fps, we write the video to 20 fps, so a 1/9 slow-down

    # camera 1 (2ms shutter speed, 3.6mm lens)
        (base) junweil@home-lab:~/projects/tennis_project$ python get_camera_images_opencv_threading_highfps.py --cam_num 0 --fps 180 --h 1024 --w 1280 --save_video --write_video_fps 20 --write_video_path /mnt/nvme1/junweil/test_videos/test_2ms_3.6mm_180to20fps.avi

    # camera 2 (2ms shutter speed, 2.8mm wide-angle lens)
        (base) junweil@home-lab:~/projects/tennis_project$ python get_camera_images_opencv_threading_highfps.py --cam_num 2 --fps 180 --h 1024 --w 1280 --save_video --write_video_fps 20 --write_video_path /mnt/nvme1/junweil/test_videos/test_2ms_2.8mm_180to20fps.avi

    # running at the same time, can get ~150 fps on a i9-10900X machine
```
3. Process the written video from avi to mp4 so we can analyze them further
```
    # watch it and find when the video should start
        (base) junweil@home-lab:/mnt/nvme1/junweil/test_videos$ mplayer test_2ms_3.6mm_180to20fps.avi

        (base) junweil@home-lab:/mnt/nvme1/junweil/test_videos$ ffmpeg -i test_2ms_3.6mm_180to20fps.avi -ss 00:00:49 -c:v libx264 test_2ms_3.6mm_180to20fps.mp4

        # and h264 video is much smaller
            (base) junweil@home-lab:/mnt/nvme1/junweil/test_videos$ du -h *
                231M    test_2ms_2.8mm_180to20fps.avi
                22M test_2ms_2.8mm_180to20fps.mp4
                158M    test_2ms_3.6mm_180to20fps.avi
                22M test_2ms_3.6mm_180to20fps.mp4
```
4. The camera's shutter speed are 2ms, we can try 0.5ms, the ball's dragging disappeared. but it is too dark. 1ms seems good enough. See this [comparison](tennis_est_examples/0.5ms_vs_1ms_vs_2ms_shutter_speed_180fpsto20fps.jpg).
With 1ms shutter speed and additional lighting, we can easily track tennis ball in this video (will need a 4090 laptop to run at 180fps. 2060 laptop run at 70fps): [this](tennis_est_examples/test_1ms_2.8mm_180fpsto20fps.gif)
