# Tongue Tip Detection and Tracking

This project is for detection of tip of tongue in real time and video, and track it.
Libraries used in this project are: Python, OpenCV, dlib, numpy, matplotlib

## Real time Demo:

<img src="https://github.com/mansikataria/tongue-tip-detection-and-tracking/blob/main/result/1.gif" width="500" height="375"/>
<img src="https://github.com/mansikataria/tongue-tip-detection-and-tracking/blob/main/result/2.gif" width="500" height="375"/>

## How to run?
### For Real time detection:
1. The default camera used is in-build webcam (0), if you re using usb webcam (1) set the camera number on line 12 in [detect-tongue-tip-real-time.py](https://github.com/mansikataria/tongue-tip-detection-and-tracking/blob/main/detect-tongue-tip-real-time.py#L12) 
2. Execute `python detect-tongue-tip-real-time.py`; press `q` to escape anytime

### For existing video
1. Get a video you want to input.
2. Change line 12 in [detect-tongue-with-blob-tracking-revised.py](https://github.com/mansikataria/tongue-tip-detection-and-tracking/blob/main/detect-tongue-with-blob-tracking-revised.py#L12) to use your video
3. Just execute `python3 detect-tongue-with-blob-tracking-revised.py`

## Overview of the logic:
1. Divide the video into frames
2. Fetch mouth part
3. Detect blob keypoints in the mouth part using OpenCV
4. Based on detections in step 2 and 3, an algorithm is implemented in python, numpy around the height of inner mouth, width of mouth, blob keypoint to detect the tip of tongue
