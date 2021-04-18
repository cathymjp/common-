from __future__ import print_function
from imutils import perspective
from imutils import contours
import argparse
import imutils
from scipy.spatial import distance as dist
import numpy as np
import cv2
from functools import reduce
import operator
import math


def order_points_old(pts):
	# initialize a list of coordinates that will be ordered such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]  # 0 original
	rect[2] = pts[np.argmax(s)]  # 2 original
	# now, compute the difference between the points, the top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]  # 1 original
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

image = cv2.imread('images/1016_15m_transL_10.jpg')
scale_percent = 20  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

green_lower = np.array([21, 80, 100], np.uint8)
green_upper = np.array([255, 255, 255], np.uint8)

green = cv2.inRange(hsv, green_lower, green_upper)
image_res = cv2.bitwise_and(image, image, mask=green)
image_res = cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB)

image_res_thre = cv2.cvtColor(image_res, cv2.COLOR_RGB2GRAY)
_, image_res_thre = cv2.threshold(image_res_thre, 255, 255, cv2.THRESH_OTSU)

# find contours in the edge map
cnts = cv2.findContours(image_res_thre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the bounding box point colors
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
coords = []
testing = []
testing2 = []
for (i, c) in enumerate(cnts):
    if cv2.contourArea(c) < 100:
        continue

    box = cv2.minAreaRect(c)
    box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    print("type", type(box))
    one = box[1]
    print("one", one)
    print("box", box)
    box = np.array(box, dtype="int")
    cv2.drawContours(image, [box], -1, (0,255,0), 2)
    print("Object #{}:".format(i+1))
    print(box)

    rect = order_points_old(box)

    print("2113213122", rect)

    # cv2.circle(image, (267, 388), 8, (10, 150, 150), -1)
    # cv2.circle(image, (390, 383), 10, (110, 15, 50), -1)
    # cv2.circle(image, (391, 514), 12, (50, 15, 250), -1)
    # cv2.circle(image, (266, 512), 14, (150, 150, 150), -1)
    print(rect.astype("int"))
    print("")
    testing.append(rect)
    coords.append(rect[0])

for((x,y), color) in zip(rect, colors):
    cv2.circle(image, (int(x), int(y)), 5, color, -1)
    cv2.putText(image, "#{}".format(i+1), (int(rect[0][0]), int(rect[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

cv2.imshow("Image", image)
# cv2.waitKey(0)
print("coords", coords)
# center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
# print(sorted(coords, key=lambda coord: (-135 - math.degrees(
#     math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
# print(np.argsort(center))
# two = rect[1]
# print("two", two)
# print("testing0", testing[0])
# print("testing1", testing[1])
# print("testing2", testing[2])
# print("testing3", testing[3])
# print("Rect", rect)
testing2.append(testing[1])
testing2.append(testing[2])
testing2.append(testing[3])
testing2.append(testing[0])
print("testing2", testing2)
cv2.circle(image, (testing2[0][0][0], testing2[0][0][1]), 8, (10, 150, 150), -1)
cv2.imshow("first corner?", image)
cv2.waitKey(0)