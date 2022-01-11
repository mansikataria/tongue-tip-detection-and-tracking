# import the necessary packages
import math
from imutils import face_utils
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
import numpy as np
import imutils
import dlib
import cv2
from scipy.spatial import distance as dist

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=False,
# 	help="path to facial landmark predictor", default="shape_predictor_68_face_landmarks.dat")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.74

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

def get_rotated_mouth_loc_with_height(image):
	x1,y1,w1,h1,h_in,y = 1,1,1,1,1,1
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	if(len(rects) > 0):
		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the landmark (x, y)-coordinates to a NumPy array
			shape = predictor(gray, rect)
			shape = shape_to_np(shape)
			x_lowest_in_face, y_lowest_in_face = shape[9]
			# loop over the face parts individually
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				if(name == "mouth"):
					# extract the ROI of the face region as a separate image
					mouth_rect = cv2.minAreaRect(np.array([shape[i:j]]))
					# mouth_box = cv2.boxPoints(rect)
					# (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
				if(name == "inner_mouth"):
					# loop over the subset of facial landmarks, drawing the
					# specific face part
					# for (x, y) in shape[i:j]:
					# 	cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
					# extract the ROI of the face region as a separate image
					inner_mouth_rect = cv2.minAreaRect(np.array([shape[i:j]]))
					# inner_mouth_box = cv2.boxPoints(rect)
		return {"mouth_rect":mouth_rect,
		"inner_mouth_rect":inner_mouth_rect, "image_ret":image, "y_lowest_in_face":y_lowest_in_face, "shape": shape}
	else:
		return {"error":"true", "message":"No Face Found!"}


def get_mouth_loc_with_height(image):
	x1,y1,w1,h1,h_in,y = 1,1,1,1,1,1
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	if(len(rects) > 0):
		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the landmark (x, y)-coordinates to a NumPy array
			shape = predictor(gray, rect)
			shape = shape_to_np(shape)
			x_lowest_in_face, y_lowest_in_face = shape[9]
			# loop over the face parts individually
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				if(name == "mouth"):
					# extract the ROI of the face region as a separate image
					(x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
				if(name == "inner_mouth"):
					# loop over the subset of facial landmarks, drawing the
					# specific face part
					# for (x, y) in shape[i:j]:
					# 	cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
					# extract the ROI of the face region as a separate image
					(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
					h_in = h
		return {"mouth_x":x1,
		"mouth_y":y1,"mouth_w":w1,"mouth_h":h1, "image_ret":image, "height_of_inner_mouth":h_in, 
		"inner_mouth_y":y, "y_lowest_in_face":y_lowest_in_face, "shape": shape}
	else:
		return {"error":"true", "message":"No Face Found!"}
							

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
			
def draw_mouth(image, shape):
	# draw mouth points
	(j, k) = FACIAL_LANDMARKS_IDXS["mouth"]
	pts_mouth = shape[j:k]
	for (x, y) in pts_mouth:
		cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
	
	# return the output image
	return image

def point_on_lower_lip(x, y, shape, x1, x2, y1, y2):
	# Convert points in resized image to those in original image
	x = math.floor((x*x1)/x2)
	y = math.floor((y*y1)/y2)
	#for each pair of locations in lower lip, create a convex hull
	lower_lip_points = [48, 59, 60, 58, 67, 57, 66, 56, 65, 55, 64, 54]
	pts = np.array([shape[i] for i in lower_lip_points])
	result = cv2.pointPolygonTest(pts, (x,y), measureDist=False)
	print(result)
	if(result>=0) :
		return True
	else:
		return False


def mouth_aspect_ratio(shape):
	# grab the indexes of the facial landmarks for the mouth
	(mStart, mEnd) = (49, 68)
	mouth = shape[mStart:mEnd] 
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar