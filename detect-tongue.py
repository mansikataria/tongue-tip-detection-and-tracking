import cv2 as cv
from imutils import face_utils
import numpy as np
import operator
import math
from matplotlib import pyplot as plt

import argparse
import imutils
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor", default="shape_predictor_68_face_landmarks.dat")
args = vars(ap.parse_args())

# The video feed is read in as
# a VideoCapture object
cap = cv.VideoCapture("test/test_vid.mp4")

# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 0
i=0
while(cap.isOpened()):
	
	# ret = a boolean return value from getting
	# the frame, frame = the current frame being
	# projected in the video
	ret, frame = cap.read()
	if not ret:
		print('No frames grabbed!')
		break
	# Opens a new window and displays the input
	# frame
	cv.imshow("input", frame)
	
	# Converts each frame to grayscale - we previously
	# only converted the first frame to grayscale
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	#get mouth part
	
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			if(name == "mouth"):
		    	# clone the original image so we can draw on it, then
		    	# display the name of the face part on the image
				clone = gray.copy()
				cv.putText(clone, name, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
		    	# loop over the subset of facial landmarks, drawing the
		    	# specific face part
				for (x, y) in shape[i:j]:
					cv.circle(clone, (x, y), 1, (0, 0, 255), -1)
            		# extract the ROI of the face region as a separate image
					(x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
					roi = gray[y:y + h, x:x + w]
					roi = imutils.resize(roi, width=250, inter=cv.INTER_CUBIC)
					print("roi shape")
					print(roi.shape)
		    		# show the particular face part
            		# cv.imshow("ROI", roi)
				break

	rects = detector(prev_gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(prev_gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			if(name == "mouth"):
		    	# clone the original image so we can draw on it, then
		    	# display the name of the face part on the image
				clone = prev_gray.copy()
				cv.putText(clone, name, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
		    	# loop over the subset of facial landmarks, drawing the
		    	# specific face part
				for (x, y) in shape[i:j]:
					cv.circle(clone, (x, y), 1, (0, 0, 255), -1)
            		# extract the ROI of the face region as a separate image
					(x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
					prev_roi = prev_gray[y:y + h, x:x + w]
					prev_roi = imutils.resize(prev_roi, width=250, inter=cv.INTER_CUBIC)
					print("prev roi shape")
					print(prev_roi.shape)
		    		# show the particular face part
            		# cv.imshow("ROI", roi)
				break

        
	# Calculates dense optical flow by Farneback method
	flow = cv.calcOpticalFlowFarneback(prev_roi, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	# Computes the magnitude and angle of the 2D vectors
	magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
	# print(magnitude)
	# print(flow.shape)
	# print(flow[:,:,0])
	# print(max(magnitude[0]))
	index, value = max(enumerate(magnitude[0]), key=operator.itemgetter(1))
	print(i)
	print(index)
	print(value)
	value = max(magnitude[0])
	cv.imwrite("frames/"+str(i)+".jpg", frame)
	# if(value > 6):
	# 	print(max(magnitude[0]))
	# 	print(magnitude)
	# 	index, value = max(enumerate(magnitude[0]), key=operator.itemgetter(1))
	# 	print(index)
	# 	# print(value)
	# 	# print(index)
	# 	color = (0, 0, 255)
	# 	image = cv.circle(frame, (math.floor(flow[index,index,0]), math.ceil(flow[index,index,1])), 5, color, thickness=-1)
	# 	cv.imwrite("frames/"+str(i)+".jpg", frame)

	# Sets image hue according to the optical flow
	# direction
	mask[..., 0] = angle * 180 / np.pi / 2
	
	# Sets image value according to the optical flow
	# magnitude (normalized)
	mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
	
	# Converts HSV to RGB (BGR) color representation
	rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
	
	# Opens a new window and displays the output frame
	i=i+1
	cv.imshow("dense optical flow", rgb)
	# Resize the Window
	# cv.resizeWindow("dense optical flow", 10, 10)
	# Updates previous frame
	prev_gray = gray
	
	# Frames are read by intervals of 1 millisecond. The
	# programs breaks out of the while loop when the
	# user presses the 'q' key
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()
