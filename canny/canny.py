import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.ndimage import convolve
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
path = 'D:\\Python\\General\\03_Projects\\01Practice\\Excercises\\Northvolt'
os.chdir(path)


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Noise Reduction

img= cv2.imread('image.jpeg')
# print(img.shape)
# transform to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Original image',img)
# cv2.imshow('Gray image', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and, then initialize the
# distance colors and reference object
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
	(255, 0, 255))
refObj = None

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 150:
		continue
	x, y, _, _ = cv2.boundingRect(c)
	print(f"x:{x}", " ",f"y:{y}")
	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	# compute the center of the bounding box
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])


	# if this is the first contour we are examining (i.e.,
	# the left-most contour), we presume this is the
	# reference object
	if refObj is None:
		# unpack the ordered bounding box, then compute the
		# midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-right and
		# bottom-right
		(tl, tr, br, bl) = box
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		# compute the Euclidean distance between the midpoints,
		# then construct the reference object
		D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		refObj = (box, (cX, cY), D / args["width"])
		continue

	# draw the contours on the image
	orig = img.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
	# stack the reference coordinates and the object coordinates
	# to include the object center
	refCoords = np.vstack([refObj[0], refObj[1]])
	objCoords = np.vstack([box, (cX, cY)])

	# loop over the original points
	for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
		# draw circles corresponding to the current points and
		# connect them with a line
		cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
		cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
		cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
			color, 2)
		cv2.line(orig, (0,int(y)),(int(x),int(y)),colors[3],2)
		cv2.line(orig, (int(x),0),(int(x),int(y)),colors[4],2)
		# compute the Euclidean distance between the coordinates,
		# and then convert the distance in pixels to distance in
		# units
		D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
		distancex = dist.euclidean((int(x),0),(int(x),int(y)))/refObj[2]
		distancey=dist.euclidean((0,int(y)),(int(x),int(y)))/refObj[2]
		(mX, mY) = midpoint((xA, yA), (xB, yB))
		(mXx, mYx) = (x, (y-0)/2)
		(mXy, mYy) = ((x-0)/2,y )
		# cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
		cv2.putText(orig, f"{D}mm", (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.putText(orig, f"{distancex}mm", (int(mXx), int(mYx - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 1, colors[4], 2)
		cv2.putText(orig, f"{distancey}mm", (int(mXy), int(mYy - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 1, colors[3], 2)	
		cv2.putText(orig, f"Inside tolerances", (20,3500),
			cv2.FONT_HERSHEY_SIMPLEX, 10, [0, 255, 0], 4)	
		# show the output image
		orig=cv2.resize(orig,(1209,1612))
		cv2.imshow("Image", orig)
		cv2.imwrite("result.jpg",orig)
		cv2.waitKey(0)
