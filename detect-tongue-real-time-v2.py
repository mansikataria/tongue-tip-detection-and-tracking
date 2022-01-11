# import the necessary packages
from imutils.video import VideoStream
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from face_utils import draw_mouth, get_mouth_loc_with_height, point_on_lower_lip

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
# time.sleep(2.0)

def checkKey(dict, key):
    return key in dict.keys()

i=0
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    i=i+1
    cv2.imwrite("frames/"+str(i)+".png", frame)
    enhanced = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
    frame = imutils.resize(frame, width=500)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # image = frame.copy()
    result = get_mouth_loc_with_height(enhanced)
    message = 'Face detected!'
    if(checkKey(result,"error")):
        message = result['message']
        cv2.imshow("input2", frame)
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 0), 1)
        continue
    else:
        mouth_x=result['mouth_x'] 
        mouth_y=result['mouth_y'] 
        mouth_w=result['mouth_w'] 
        mouth_h=result['mouth_h'] 
        image_ret=result['image_ret'] 
        height_of_inner_mouth=result['height_of_inner_mouth'] 
        inner_mouth_y=result['inner_mouth_y'] 
        y_lowest_in_face=result['y_lowest_in_face']
        shape=result['shape']
        frame = draw_mouth(frame, shape)
        #Cut upper lip part
        # mouth_h = height_of_inner_mouth + abs(mouth_y-inner_mouth_y)
        # mouth_h = mouth_h+16

        roi = image_ret[inner_mouth_y:y_lowest_in_face, mouth_x:mouth_x + mouth_w]
        x1,y1,c1 = roi.shape
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        x2,y2,c2 = roi.shape
        # get lowest blob point in mouth
        # dst = cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
        # cv2.imshow('enhanced', dst)

        #detect blob
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(roi,None)
                
        # compute the descriptors with ORB
        kp, des = orb.compute(roi, kp)
        x_fin,y_fin =1,1
        if(kp):
            x_fin,y_fin= kp[0].pt
            for point in kp:
                x,y= point.pt
                # if(x_fin<x):
                #     x_fin = x
                # and not point_on_lower_lip(x, y, shape)
                if(y_fin<y) :
                    y_fin = y
                    x_fin = x
        
        print(x_fin,y_fin)

        #convert
        x_fin = math.floor((x_fin*x1)/x2)
        y_fin = math.floor((y_fin*y1)/y2)
        # print(x_fin,y_fin)
        
        # if(y_fin<=1):
        #     cv2.imwrite("defective_frames/"+str(i)+".jpg", image_ret)
        cv2.putText(frame, "inner mouth height: " + str(height_of_inner_mouth), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 1)
        cv2.putText(frame, "tongue tip location: " + str(x_fin) + "," + str(y_fin), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 1)
        
                # (abs(height_of_inner_mouth-y_fin) > 8 and
                #  and (y_fin>19.6 and x_fin < 40)
        # Based on experimentation, mouth is open if inner_mouth_height >8
        # If y_fin<=1 , then no blob keypoint is found
        if(height_of_inner_mouth>=8 and y_fin>1) :
            # if(y_fin > 14) :
                color = (0, 0, 255)
                # print(image_ret[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w].shape)
                # print()
                image = cv2.circle(frame[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (x_fin, y_fin), 4, color, thickness=-1)
                cv2.imshow("input2", frame)
                # cv2.imwrite("frames/"+str(i)+".jpg", frame)
        else:
            cv2.imshow("input2", frame)
        
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()