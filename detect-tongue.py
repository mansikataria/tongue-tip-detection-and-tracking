import cv2 as cv
import imutils
import numpy as np
import operator
import math
from matplotlib import pyplot as plt
from numpy.core.numeric import Inf

from face_utils import get_mouth_loc

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
    mouth = np.zeros_like(frame)
    mouth[180:260,250:370]=frame[180:260,250:370]
    
    # cv.imshow("input", frame)
    # cv.imshow("input2", mouth)
    
    i=i+1
    
    # cv.imshow("Frame", frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    # gray = cv.cvtColor(mouth, cv.COLOR_BGR2GRAY) 
    image = frame.copy()
    mouth_x, mouth_y, mouth_w, mouth_h, image_ret = get_mouth_loc(image)
    # print(mouth_x, mouth_y, mouth_w, mouth_h)
    roi = image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]
    roi = imutils.resize(roi, width=250, inter=cv.INTER_CUBIC)
	#  show the particular face part
    # cv.imshow("ROI", roi)
    gray = cv.cvtColor(image_ret, cv.COLOR_BGR2GRAY) 
    flow = cv.calcOpticalFlowFarneback(prev_gray[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], 
        gray[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w],None,0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # # plt.imshow(magnitude)
    # # plt.show()
    # # print(magnitude.shape)
    # # print(angle.shape)
    # # print(flow.shape)
    minv, maxv, minl, maxl = cv.minMaxLoc(magnitude)
    # index, value = max(enumerate(magnitude[0]), key=operator.itemgetter(1))
    print(maxv)
    print(i)
    
    color = (0, 0, 255)
    if(maxv >= 2.3 and maxv <=6):
        max_mag = maxl
        # print(index)
        image = cv.circle(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], max_mag, 5, color, thickness=-1)
        cv.imshow("input2", image_ret)
        cv.imwrite("frames/"+str(i)+".jpg", image)
    else:
        if(max_mag != 1):
            image = cv.circle(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], max_mag, 5, color, thickness=-1)
        cv.imshow("input2", image_ret)


	    # direction
        mask[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w, 0] = angle * 180 / np.pi / 2
        mask[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w, 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
	
	    # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", rgb)
    
    prev_gray = gray
	
	# user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    
# The following frees up resources and
# cap.release()
# cv.destroyAllWindows()