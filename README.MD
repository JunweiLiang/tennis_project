# A simple Tennis Tracking/Robot Coach project for undergrads

## Introduction

This repo contains clean exercise code for:

+ grab images from laptop cameras
+ running yolo object detection on web cameras
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
