# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])

# cv2.imshow('enhanced', dst)
alpha = 1.7 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
# adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
dst = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
image = imutils.resize(dst, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)


if(len(rects) > 0) :
    x1,x2,y1,y2,h1,h2,w1,w2 = 1,1,1,1,1,1,1,1

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        face = face_utils.visualize_facial_landmarks(image, shape)
        cv2.imshow("face", face)
        
        #Mask the lips
        # masked_image = mask_the_lips(image, shape)
        # cv2.imshow("masked_image",masked_image)
        
        # loop over the face parts individually

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if(name == "inner_mouth"):
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                # extract the ROI of the face region as a separate image
                (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = image[y1:y1 + h1, x1:x1 + w1]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                # show the particular face part
                cv2.imshow("inner mouth", roi)
                print("height of inner mouth:")
                print(h1)

            if(name == "mouth"):
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                # extract the ROI of the face region as a separate image
                (x2, y2, w2, h2) = cv2.boundingRect(np.array([shape[i:j]]))
                # print("height")
                # print(h)

        # cut the upper lip part
        # print("Height of mouth")
        # print(h2)
        # print(abs(y1-y2))
        # h2 = h1 + abs(y1-y2)
        # print("New h2")
        # print(h2)
        # print(y2+h2)
        # print(shape[9])
        x_lowest_in_face, y_lowest_in_face = shape[9]
        # roi = image[y2:y2 + h2+16, x2:x2 + w2]
        # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        # show the particular face part
        # cv2.imshow("ROI1", roi)
        roi = image[y1:y_lowest_in_face, x2:x2 + w2]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        # show the particular face part
        cv2.imshow("mouth with increased height", roi)

        # # Find Canny edges
        # edged = cv2.Canny(roi, 30, 200)
        # # find contours
        # # ret, thresh = cv2.threshold(roi, 127, 255, 0)
        # contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # cv2.drawContours(roi, contours, -1, (0,255,0), 3)
        # cv2.imshow('Contours', roi)
        
        dst = cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
        # cv2.imshow('enhanced', dst)
        #detect blob
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create()

        # Detect blobs.
        keypoints = detector.detect(dst)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(dst, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)

        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(dst,None)
        
        # compute the descriptors with ORB
        kp, des = orb.compute(dst, kp)
        # kp_new = []
        # x_fin,y_fin= kp[0].pt
        # for i in range(len(kp)):
        #     point = kp[i]
        #     d=des[i]
        #     x,y= point.pt
        #     print(x, y)
        #     print(d)
        #     if(dst[x,y] != 0):
        #         kp_new.append(point)
        #     if(y_fin<y):
        #         y_fin = y

        # print(x_fin,y_fin)
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(dst, kp, None, color=(0,255,0), flags=0)
        plt.imshow(img2), plt.show()    

else:
    print("No Face detected")