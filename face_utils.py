# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor", default="shape_predictor_68_face_landmarks.dat")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def get_mouth_loc_with_height(image):
	x,y,w,h,h_in = 1,1,1,1,1
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)
		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			if(name == "mouth"):
		    	# clone the original image so we can draw on it, then
		    	# display the name of the face part on the image
				clone = image.copy()
				cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
		    	# loop over the subset of facial landmarks, drawing the
		    	# specific face part
				for (x, y) in shape[i:j]:
					cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            	# extract the ROI of the face region as a separate image
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				# roi = image[y:y + h, x:x + w]
				# roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		    	# show the particular face part
				# cv2.imshow("ROI-1", roi)
				# print(x, y, w , h)
				# return x,y,w,h, image
			if(name == "inner_mouth"):
		    	# loop over the subset of facial landmarks, drawing the
		    	# specific face part
				# for (x, y) in shape[i:j]:
				# 	cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            	# extract the ROI of the face region as a separate image
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				h_in = h	
	return x,y,w,h, image, h_in

def get_mouth_loc(image):
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)
		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			if(name == "mouth"):
		    	# clone the original image so we can draw on it, then
		    	# display the name of the face part on the image
				clone = image.copy()
				cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
		    	# loop over the subset of facial landmarks, drawing the
		    	# specific face part
				for (x, y) in shape[i:j]:
					cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            	# extract the ROI of the face region as a separate image
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				# roi = image[y:y + h, x:x + w]
				# roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		    	# show the particular face part
				# cv2.imshow("ROI-1", roi)
				# print(x, y, w , h)
				return x,y,w,h, image
			