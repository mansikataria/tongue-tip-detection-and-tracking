# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from face_utils import get_mouth_loc_with_height, shape_to_np

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = frame.copy()
    mouth_x, mouth_y, mouth_w, mouth_h, image_ret, height_of_inner_mouth, inner_mouth_y, y_lowest_in_face = get_mouth_loc_with_height(image)
    mouth_h = mouth_h+16
    roi = image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]
    x1,y1,c1 = roi.shape
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    x2,y2,c2 = roi.shape
    # get lowest blob point in mouth
    dst = cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
    # cv2.imshow('enhanced', dst)

    #detect blob
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(dst,None)
            
    # compute the descriptors with ORB
    kp, des = orb.compute(dst, kp)
    x_fin,y_fin =1,1
    if(kp):
        x_fin,y_fin= kp[0].pt
        for point in kp:
            x,y= point.pt
            # if(x_fin<x):
            #     x_fin = x
            if(y_fin<y):
                y_fin = y
                x_fin = x
    
    print(x_fin,y_fin)

    #convert
    x_fin = (x_fin*x1)/x2
    y_fin = (y_fin*y1)/y2
    print(x_fin,y_fin)
    
    cv2.putText(image_ret, "inner mouth height: " + str(height_of_inner_mouth), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
    cv2.putText(image_ret, "blob: " + str(x_fin) + "," + str(y_fin), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
    if(abs(height_of_inner_mouth-y_fin) > 9 or (y_fin>19.6 and x_fin < 40)) :
        # if(y_fin > 14) :
            color = (0, 0, 255)
            print(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w].shape)
            # print()
            image = cv2.circle(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (math.floor(x_fin), math.ceil(y_fin)), 4, color, thickness=-1)
            cv2.imshow("input2", image_ret)
            # cv2.imwrite("frames/"+str(i)+".jpg", image_ret)
    else:
        cv2.imshow("input2", image_ret)
      
    # show the frame
    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()