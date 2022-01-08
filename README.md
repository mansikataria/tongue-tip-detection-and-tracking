# Tongue Tip Detection and Tracking

This project is for detection of tip of tongue in video, and track it.
Libraries used in this project are: Python, OpenCV, dlib, numpy, matplotlib

## Demo:

![Input](https://github.com/mansikataria/tongue-tip-detection-and-tracking/blob/main/test/test_vid.gif)  
![Output](https://github.com/mansikataria/tongue-tip-detection-and-tracking/blob/main/result/test-vid.gif)

## How to run?
1. Get a video you want to input.
2. Change line 12 in [detect-tongue-with-blob-tracking-revised.py](https://github.com/mansikataria/tongue-tip-detection-and-tracking/blob/main/detect-tongue-with-blob-tracking-revised.py#L12) to use your video
3. Just execute `python3 detect-tongue-with-blob-tracking-revised.py`

## Overview of the logic:
1. Divide the video into frames
2. Fetch mouth part
3. Detect blob keypoints in the mouth part using OpenCV
4. Write a logic in python, numpy around the height of inner mouth, width of mouth, blob keypoint to detect the tip of tongue

