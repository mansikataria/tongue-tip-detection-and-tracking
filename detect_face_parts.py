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

dst = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
# cv2.imshow('enhanced', dst)

#detect blob
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(dst,None)
# compute the descriptors with ORB
kp, des = orb.compute(dst, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(dst, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()

image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

x1,x2,y1,y2,h1,h2,w1,w2 = 1,1,1,1,1,1,1,1

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
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
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		    # show the particular face part
            cv2.imshow("ROI", roi)
            print("height")
            print(h)
            # if(h > 10):
            #     dst = cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
            #     cv2.imshow('enhanced', dst)
            #     #detect blob
            #     # Initiate ORB detector
            #     orb = cv2.ORB_create()
            #     # find the keypoints with ORB
            #     kp = orb.detect(dst,None)
            #     # compute the descriptors with ORB
            #     kp, des = orb.compute(dst, kp)
            #     # draw only keypoints location,not size and orientation
            #     img2 = cv2.drawKeypoints(dst, kp, None, color=(0,255,0), flags=0)
            #     plt.imshow(img2), plt.show()
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
            (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
            # print("height")
            # print(h)
            print()
            roi = image[y1:y1 + h1, x1:x1 + w1]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		    # show the particular face part
            cv2.imshow("ROI1", roi)
            roi = image[y1:y1 + h1+16, x1:x1 + w1]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		    # show the particular face part
            cv2.imshow("ROI2", roi)

            # image=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            # se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
            # bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
            # out_gray=cv2.divide(image, bg, scale=255)
            # out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 
            # cv2.imshow('out_binary', out_binary)

            dst = cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
            # cv2.imshow('enhanced', dst)
            #detect blob
            # Initiate ORB detector
            orb = cv2.ORB_create()
            # find the keypoints with ORB
            kp = orb.detect(dst,None)
            
            # compute the descriptors with ORB
            kp, des = orb.compute(dst, kp)
            x_fin,y_fin= kp[0].pt
            for point in kp:
                x,y= point.pt
                if(x_fin<x):
                    x_fin = x
                if(y_fin<y):
                    y_fin = y
            print(x_fin,y_fin)
            print(y1+h1)
            # draw only keypoints location,not size and orientation
            img2 = cv2.drawKeypoints(dst, kp, None, color=(0,255,0), flags=0)
            plt.imshow(img2), plt.show()
         
        if(name == "jaw"):
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
           

            # increase contrast
            # alpha = 1.5 # Contrast control (1.0-3.0)
            # beta = 0 # Brightness control (0-100)
            # adjusted = cv2.convertScaleAbs(dst, alpha=alpha, beta=beta)
            # cv2.imshow('Contrast', adjusted)

            # # # find contours
            # ret, thresh = cv2.threshold(adjusted, 127, 255, 0)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            # cv2.drawContours(adjusted, contours, -1, (0,255,0), 3)
            # cv2.imshow('Contours', adjusted)

            # Get histogram of particular face part
            # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            # # compute a grayscale histogram
            # hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
            # # print(hist)
            # # matplotlib expects RGB images so convert and then display the image
            # # with matplotlib
            # plt.figure()
            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB))
            # # plot the histogram
            # plt.figure()
            # plt.title("Grayscale Histogram")
            # plt.xlabel("Bins")
            # plt.ylabel("# of Pixels")
            # plt.plot(hist)
            # plt.xlim([0, 256])

            # # normalize the histogram
            # hist /= hist.sum()
            # # plot the normalized histogram
            # plt.figure()
            # plt.title("Grayscale Histogram (Normalized)")
            # plt.xlabel("Bins")
            # plt.ylabel("% of Pixels")
            # plt.plot(hist)
            # fig_name = name + "_" + args["image"].split("/")[1]
            # plt.savefig(fig_name)
            # plt.xlim([0, 256])
            # plt.show()

            # print(hist.mean())

            # edges = cv2.Canny(dst,50,300)
            # print(len(edges))
            # result = "No Tongue"
            # if(len(edges)>100):
            #     result = "Tongue"
            # plt.subplot(121),plt.imshow(dst,cmap = 'gray')
            # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            # plt.show()

            # cv2.imshow("Image", clone)
            # cv2.waitKey(0)



# visualize all facial landmarks with a transparent overlay
# output = face_utils.visualize_facial_landmarks(image, shape)
# cv2.imshow("Image", output)
cv2.waitKey(0)