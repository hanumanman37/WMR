# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import time
import math
import serial
import configparser

#
def midpoint(ptA, ptB):
	return	((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def pos(a,b,c):
	phi = (a**2 + c**2 - b**2)/ (2*a*c)
	x = a * np.cos(phi)
	y = a * np.sin(phi)
	return (x,y)
#IP camera setup
url = "http://192.168.1.45:8080"
cap = cv2.VideoCapture(url+"/video")
#cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
#ref width
width = 10 #cm
#warmup cam
time.sleep(2.0)
#Green filter
greenLower = (63, 94, 108)
greenUpper = (87, 255, 255)
# Cam info
print("Opened source. FPS: {}, width: {}, height: {}".format(
	int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Implement a frames per second counter
#start_time = time.time()
#i = 0
count = 0
while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		print("frame read failed")
		break

	#i += 1
	#time_diff = time.time() - start_time
	#if time_diff != 0:
	#	print("FPS: {}\n".format(round(i / time_diff, 2)), end='')

	#image preprocessing
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.inRange(hsv, greenLower, greenUpper)
	edged = cv2.erode(edged, None, iterations=1)
	edged = cv2.dilate(edged, None, iterations=1)
	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# sort the contours from left-to-right and, then initialize the
	# distance colors and reference object
	if len(cnts) > 0:
		(cnts, _) = contours.sort_contours(cnts)
	colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
	refObj = None
	# loop over the contours individually
	for c in cnts:
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 100:
			continue
		# compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.boxPoints(box)
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
			refObj = (box, (cX, cY), D / width)
			continue
		# draw the contours on the image
		orig = frame.copy()
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
		# stack the reference coordinates and the object coordinates
		# to include the object center
		refCoords = np.vstack([refObj[0], refObj[1]])
		objCoords = np.vstack([box, (cX, cY)])
		coord=[]
		# loop over the original points
		for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):

			cv2.line(orig, (int(refCoords[4][0]), int(refCoords[4][1])), (int(objCoords[4][0]), int(objCoords[4][1])), color, 1)
			distanceCornerToCenter = (dist.euclidean((xA, yA), (objCoords[4])) / refObj[2])
			cv2.line(orig, (int(refCoords[0][0]), int(refCoords[0][1])), (int(objCoords[4][0]), int(objCoords[4][1])),
					 (0, 255, 255), 2)
			cv2.line(orig, (int(refCoords[1][0]), int(refCoords[1][1])), (int(objCoords[4][0]), int(objCoords[4][1])),
					 (0, 255, 255), 2)
			D1 = (dist.euclidean((refCoords[0][0], refCoords[0][1]), (objCoords[4][0], objCoords[4][1]))) / refObj[2]
			D2 = (dist.euclidean((refCoords[1][0], refCoords[1][1]), (objCoords[4][0], objCoords[4][1]))) / refObj[2]

			coord.append(distanceCornerToCenter)
			(mX, mY) = midpoint((xA, yA), (xB, yB))
			# show the output image
			cv2.imshow("Image", orig)
			if (len(coord) > 2):
				cPos = pos(coord[0], coord[1], 10)
				cv2.putText(orig, f"x:{round(cPos[0], 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0),
							1)
				cv2.putText(orig, f"y:{round(cPos[1], 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0),
							1)
				print(cPos)
				# X coordinate
				x1 = math.modf(cPos[0])
				xi = int(x1[1])
				xd = int(x1[0]*100)
				# Y coordinate
				y1 = math.modf(cPos[1])
				yi = int(y1[1])
				yd = int(y1[0]*100)
				# convert to byte data type
				xis = xi.to_bytes(1,'little')
				xds = xd.to_bytes(1,'little')
				yis = yi.to_bytes(1, 'little')
				yds = yd.to_bytes(1, 'little')
				count = count + 1
				if count == 1000:
					with serial.Serial('com2',115200) as ser:
						ser.write(xis)
						ser.write(xds)
						ser.write(yis)
						ser.write(yds)
						ser.close()
					print(cPos[0],cPos[1])

					count = 0

			cv2.waitKey(1) & 0xFF == ord('q')
cap.release()
cv2.destroyAllWindows()
