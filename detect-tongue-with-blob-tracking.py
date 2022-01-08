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
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255


i = 0
max_mag = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    
    #new
    # mouth = np.zeros_like(frame)
    # mouth[180:260,250:370]=frame[180:260,250:370]
    
    # cv.imshow("input", frame)
    # cv.imshow("input2", mouth)
    
    i=i+1
    print("frame: "+ str(i))
    # cv.imshow("Frame", frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    # gray = cv.cvtColor(mouth, cv.COLOR_BGR2GRAY) 
    image = frame.copy()
    mouth_x, mouth_y, mouth_w, mouth_h, image_ret, height_of_inner_mouth = get_mouth_loc_with_height(image)
    mouth_h = mouth_h+16
    # print(mouth_x, mouth_y, mouth_w, mouth_h)
    roi = image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]
    # print(roi.shape)
    x1,y1,c1 = roi.shape
    roi = imutils.resize(roi, width=250, inter=cv.INTER_CUBIC)
    # print(roi.shape)
    x2,y2,c2 = roi.shape
    #  show the particular face part
    # cv.imshow("ROI", roi)
    
    print("height_of_inner_mouth")
    print(height_of_inner_mouth)

    # get lowest blob point in mouth
    dst = cv.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
    # cv2.imshow('enhanced', dst)
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
    
    # gray = cv.cvtColor(image_ret, cv.COLOR_BGR2GRAY) 
    # flow = cv.calcOpticalFlowFarneback(prev_gray[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], 
    #     gray[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w],None,0.5, 3, 15, 3, 5, 1.2, 0)

    # # Computes the magnitude and angle of the 2D vectors
    # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # # plt.imshow(magnitude)
    # # plt.show()
    # # print(magnitude.shape)
    # # print(angle.shape)
    # # print(flow.shape)
    # minv, maxv, minl, maxl = cv.minMaxLoc(magnitude)
    # index, value = max(enumerate(magnitude[0]), key=operator.itemgetter(1))
    # print(maxv)
    # print(i)
     #Logic
    cv.putText(image_ret, "inner mouth height: " + str(height_of_inner_mouth), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
    cv.putText(image_ret, "blob: " + str(x_fin) + "," + str(y_fin), (10, 60), cv.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
    if(height_of_inner_mouth > 8) :
        if(y_fin > 14) :
            color = (0, 0, 255)
            print(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w].shape)
            # print()
            image = cv.circle(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (math.floor(x_fin), math.ceil(y_fin)), 4, color, thickness=-1)
            cv.imshow("input2", image_ret)
            cv.imwrite("frames/"+str(i)+".jpg", image_ret)
            # cv.imwrite("frames/"+str(i)+"-no-dot.jpg", image_ret)
        else:
            cv.imshow("input2", image_ret)
            cv.imwrite("frames/"+str(i)+".jpg", image_ret)
            # cv.imwrite("frames/"+str(i)+"-no-dot.jpg", image_ret)
    else:
        cv.imshow("input2", image_ret)
        cv.imwrite("frames/"+str(i)+".jpg", image_ret)
        # cv.imwrite("frames/"+str(i)+"-no-dot.jpg", image_ret)
            # if(y_fin != 1):
            #     image = cv.circle(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (x_fin, y_fin), 4, color, thickness=-1)
            # cv.imshow("input2", image_ret)


	    # direction
        # mask[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w, 0] = angle * 180 / np.pi / 2
        # mask[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w, 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
	
	    # # Opens a new window and displays the output frame
        # cv.imshow("dense optical flow", rgb)
    
    prev_gray = gray
	
	# user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    
# The following frees up resources and
# cap.release()
# cv.destroyAllWindows()