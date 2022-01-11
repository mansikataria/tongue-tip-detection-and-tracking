import cv2 as cv
import imutils
import numpy as np
import operator
import math
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.core.numeric import Inf

from face_utils import get_mouth_loc_with_height

cap = cv.VideoCapture("test/test_vid.mp4")
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('output.avi', -1, fourcc, size)


i = 0
max_mag = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    i=i+1
    print("frame: "+ str(i))
    image = frame.copy()
    mouth_x, mouth_y, mouth_w, mouth_h, image_ret, height_of_inner_mouth = get_mouth_loc_with_height(image)
    mouth_h = mouth_h+16
    roi = image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]
    x1,y1,c1 = roi.shape
    roi = imutils.resize(roi, width=250, inter=cv.INTER_CUBIC)
    x2,y2,c2 = roi.shape
    # get lowest blob point in mouth
    dst = cv.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
    # cv.imshow('enhanced', dst)

    #detect blob
    # Initiate ORB detector
    orb = cv.ORB_create()
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
    
    cv.putText(image_ret, "inner mouth height: " + str(height_of_inner_mouth), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
    cv.putText(image_ret, "blob: " + str(x_fin) + "," + str(y_fin), (10, 60), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
    if(abs(height_of_inner_mouth-y_fin) > 9 or (y_fin>19.6 and x_fin < 40)) :
        # if(y_fin > 14) :
            color = (0, 0, 255)
            print(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w].shape)
            # print()
            image = cv.circle(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (math.floor(x_fin), math.ceil(y_fin)), 4, color, thickness=-1)
            cv.imshow("input2", image_ret)
            # cv.imwrite("frames/"+str(i)+".jpg", image_ret)
    else:
        cv.imshow("input2", image_ret)
        # cv.imwrite("frames/"+str(i)+".jpg", image_ret)
        # cv.imwrite("frames/"+str(i)+"-no-dot.jpg", image_ret)
            # if(y_fin != 1):
            #     image = cv.circle(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (x_fin, y_fin), 4, color, thickness=-1)
            # cv.imshow("input2", image_ret)

    # out.write(image_ret)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    
# The following frees up resources and
# out.release()
cap.release()
cv.destroyAllWindows()