import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
from scipy.spatial import distance as dist
from imutils import contours
from imutils import perspective
from math import atan2,degrees
import statistics


def resize_image(image):
    scale_percent = 17  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    print("width", width)
    height = int(image.shape[0] * scale_percent / 100)
    print("height", height)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Resized image", image)
    return image


def apply_threshold(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_lower = np.array([21, 80, 0], np.uint8)
    green_upper = np.array([255, 255, 255], np.uint8)

    green = cv2.inRange(hsv, green_lower, green_upper)
    image_res1 = cv2.bitwise_and(image, image, mask=green)
    image_res = cv2.cvtColor(image_res1, cv2.COLOR_BGR2RGB)

    image_res_thre = cv2.cvtColor(image_res, cv2.COLOR_RGB2GRAY)
    _, image_res_thre = cv2.threshold(image_res_thre, 255, 255, cv2.THRESH_OTSU)

    return image_res1, image_res_thre


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    # angle1 = (math.atan2(p3y - p1y, p3x - p1x) - math.atan2(p2y - p1y, p2x - p1x))

    o1 = math.atan2((a[1] - b[1]), (a[0] - b[0]))
    o2 = math.atan2((c[1] - b[1]), (c[0] - b[0]))

    # abs((o1 - o2) * 180 / math.pi)
    # print("tesssint", abs((o1 - o2) * 180 / math.pi))
    return ang + 360 if ang < 0 else ang
    # return abs((o1 - o2)) * 180 / math.pi


def midpoint(x, y):
    return (x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5


def calculate_distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def distance_to_camera(known_width, focal_length, pic_width):
    return (known_width * focal_length) / pic_width


def angle_change(initial, moved):
    x_change = moved[0] - initial[0]
    y_change = moved[1] - initial[1]
    return degrees(atan2(y_change, x_change))


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

    # print("pts", pts)
    # ptsarray = [pts]
    # print("ptssaary", ptsarray)
    # # first_corners = pts[:len(pts) // 2]
    # rect = np.zeros((4, 2), dtype="float32")
    # # top-left point will have smallest sum, whereas bottom-right point will have largest sum
    # # s = pts.sum(axis=1)
    # # print("s", s)
    # # rect[0] = pts[np.argmin(s)]
    # # print("rect[0]", rect[0])
    # # top = tuple(pts[pts[:, :, 1].argmin()][0])
    #
    # rect[0] = tuple(pts[pts[:, :, 0].argmin()][0])
    #
    # rect[2] = tuple(pts[pts[:, :, 0].argmax()][0])
    # rect[1] = tuple(pts[pts[:, :, 1].argmin()][0])
    # rect[3] = tuple(pts[pts[:, :, 1].argmax()][0])
    # print("top-left", rect[0])
    # print("top-right", rect[1])
    # print("bottom-right", rect[2])
    # print("bottom-left", rect[3])
    # # cv2.circle(image, tuple(rect[0]), 1, (100, 100, 100), thickness=2, lineType=8, shift=0)
    # # cv2.circle(image, tuple(rect[1]), 1, (255, 0, 0), thickness=2, lineType=8, shift=0)
    # #
    # # cv2.circle(image, tuple(rect[2]), 1, (255, 255, 0), thickness=2, lineType=8, shift=0)
    # # cv2.circle(image, tuple(rect[3]), 1, (255, 0, 255), thickness=2, lineType=8, shift=0)
    # # cv2.imshow("circles", image)
    # # print("np.argmax(s)", pts[np.argmax(s)])
    # # rect[2] = pts[np.argmax(s)]
    # # now, compute the difference between the points, the
    # # top-right point will have the smallest difference,
    # # whereas the bottom-left will have the largest difference
    # # diff = np.diff(pts, axis=1)
    # # rect[1] = pts[np.argmin(diff)]
    # # rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def order_points(pts):
    # print("pts`````", pts)
    # top-left, top-right, bottom-right, and bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    # print("s'''''", s)

    rect[0] = pts[np.argmin(s)]  # 0 original
    rect[2] = pts[np.argmax(s)]  # 2 original

    # compute the difference between points, the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    # print("diff", diff)
    rect[1] = pts[np.argmin(diff)]  # 1 original
    rect[3] = pts[np.argmax(diff)]  # 3 original
    # print("np.argmin(s)", pts[np.argmin(s[0])])
    # print("rect[0]", rect[0])
    # print("np.argmax(s)", pts[np.argmax(s)])
    # print("Rrrrrrrr", rect)
    return rect


def compare_lists(list_to_compare):
    # print("list_to_compare", list_to_compare)
    # print("list_to_compare[0][0][0]", list_to_compare[0][0][0])
    # print("list_to_compare[0][1][0]", list_to_compare[0][1][0])
    # print("list_to_compare[0][2][0]", list_to_compare[0][2][0])
    # print("list_to_compare[0][3][0]", list_to_compare[0][3][0])
    one = list_to_compare[0][0][0] + list_to_compare[0][0][1]
    two = list_to_compare[0][1][0] + list_to_compare[0][1][1]
    three = list_to_compare[0][2][0] + list_to_compare[0][2][1]
    four = list_to_compare[0][3][0] + list_to_compare[0][3][1]
    # print("one", one)
    # print("two", two)
    # print("three", three)
    # print("four", four)

    if (one <= two) and (one <= three) and (one <= four):
        print("First")
        first = 0
        # print("first true")
        if (four >= two) and (four >= three):
            fourth = 3
            if two >= three:
                second = 2
                third = 1
            else:
                second = 1
                third = 2
        elif (three >= two) and (three >= four):
            third = 3
            if two >= four:
                second = 2
                fourth = 1
            else:
                second = 1
                fourth = 2
        elif (two >= three) and (two >= four):
            second = 3
            if four >= three:
                fourth = 2
                third = 1
            else:
                fourth = 1
                third = 2
    elif (two <= one) and (two <= three) and (two <= four):
        print("Second")
        second = 0
        # print("second true")
        if (four >= one) and (four >= three):
            print("one")
            fourth = 3
            if one >= three:
                first = 2
                third = 1
            else:
                first = 1
                third = 2
        elif (three >= one) and (three >= four):
            print("two")
            third = 3
            if one >= four:
                first = 2
                fourth = 1
            else:
                first = 1
                fourth = 2
        elif (one >= three) and (one >= four):
            print("three")
            first = 3
            if four >= three:
                fourth = 2
                third = 1
            else:
                fourth = 1
                third = 2
    elif (three <= two) and (three <= one) and (three <= four):
        third = 0
        if (four >= two) and (four >= one):
            fourth = 3
            if two >= one:
                second = 2
                first = 1
            else:
                second = 1
                first = 2
        elif (one >= two) and (one >= four):
            first = 3
            if two >= four:
                second = 2
                fourth = 1
            else:
                second = 1
                fourth = 2
        elif (two >= one) and (two >= four):
            second = 3
            if four >= one:
                fourth = 2
                first = 1
            else:
                fourth = 1
                first = 2
    elif (four <= two) and (four <= one) and (four <= three):
        print("Fourth")
        fourth = 0
        if (three >= two) and (three >= one):
            third = 3
            if two >= one:
                second = 2
                first = 1
            else:
                second = 1
                first = 2
        elif (one >= two) and (one >= three):
            first = 3
            if two >= four:
                second = 2
                third = 1
            else:
                second = 1
                third = 2
        elif (two >= three) and (two >= one):
            second = 3
            if one >= three:
                third = 1
                first = 2
            else:
                third = 2
                first = 1
    # print("first second third fourth: ", first, second, third, fourth)
    # if (list_to_compare[0][0][0] >= list_to_compare[0][1][0]) and (list_to_compare[0][0][0] >= list_to_compare[0][2][0]):
    #     # print("first true")
    #     first = 1
    #     if list_to_compare[0][1][1] > list_to_compare[0][2][1]:
    #         third = 0
    #         second = 2
    #     else:
    #         third = 2
    #         second = 0
    # elif list_to_compare[0][1][0] >= list_to_compare[0][0][0] and list_to_compare[0][1][0] >= list_to_compare[0][2][0]:
    #     # print("second true")
    #     second = 1
    #     if list_to_compare[0][0][1] > list_to_compare[0][2][1]:
    #         third = 0
    #         first = 2
    #         # print("yes-1")
    #     else:
    #         third = 2
    #         first = 0
    #         # print("yes-2")
    # elif list_to_compare[0][2][0] >= list_to_compare[0][0][0] and list_to_compare[0][2][0] >= list_to_compare[0][1][0]:
    #     # print("third true")
    #     third = 1
    #     if list_to_compare[0][0][1] > list_to_compare[0][1][1]:
    #         second = 0
    #         first = 2
    #     else:
    #         second = 2
    #         first = 0
    print("first second third fourth: ", first, second, third, fourth)

    return first, second, third, fourth


def grab_contour(threshold_image):
    # sort the contours from left-to-right and initialize the 'pixels per metric' calibration variable
    # edge = cv2.Canny(threshold_image, 75, 200)
    cnts = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the bounding box point colors
    (cnts, _) = contours.sort_contours(cnts)
    new_swap_list = []
    leftmost_contour = None

    center_points, areas, distances, corners, three_areas = [], [], [], [], []
    coords = []
    testing = []
    testing2 = []
    known_width = 7.6
    focal_length = 300
    for (i, c) in enumerate(cnts):
        area = cv2.contourArea(c)
        three_areas.append(area)
        sorteddata = sorted(zip(three_areas, cnts), key=lambda x: x[0], reverse=True)
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        one = box[1]
        box = np.array(box, dtype="int")
        cv2.drawContours(threshold_image, [box], -1, (0, 255, 0), 2)
        rect = order_points_old(box)

        testing.append(rect)
        coords.append(rect[0])

    tessss = []
    # Four largest contours' coordinates
    compare_list = [sorteddata[0][1][0][0], sorteddata[1][1][0][0], sorteddata[2][1][0][0], sorteddata[3][1][0][0]]
    first, second, third, fourth = compare_lists([compare_list])
    tessss.append(first)
    tessss.append(second)
    tessss.append(third)
    tessss.append(fourth)

    for i in np.argsort(tessss):
        new_swap_list.append(sorteddata[i][1])

    for c in new_swap_list:
        # print("c", c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        area = cv2.contourArea(c)

        if cv2.contourArea(c) < 100:
            continue

        box = approx
        box = np.squeeze(box)

        # order the points in the contour and draw outlines of the rotated rounding box
        box = order_points(box)
        box = perspective.order_points(box)
        # print("box 1111", box)
        testing.append(box)

        (x, y, w, h) = cv2.boundingRect(c)

        # compute area
        area = cv2.contourArea(c)
        areas.append(area)

        # compute center points
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        center = (cx, cy)
        center_points.append(center)

        c_x = np.average(box[:, 0])
        c_y = np.average(box[:, 1])

        # compute corners from contour image
        # four_corners = corners_from_contour(threshold_image, c)
        corners.append(box)
        # print("corners", corners)

        # compute and return the distance from the maker to the camera
        distances.append(distance_to_camera(known_width, focal_length, w))

        if leftmost_contour is None:
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the Euclidean distance between the midpoints, then construct the reference object
            d = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            leftmost_contour = (box, (c_x, c_y), d / 7.5)
            # first_box = box
            continue
    # print("center points", center_points)
    # print("cetnerpoints[1][0]", center_points[1][0])
    if center_points[1][0] <= center_points[2][0]:
        tmp = center_points[2]
        center_points[2] = center_points[1]
        center_points[1] = tmp

    print("testing distance to camera 2:", distance_to_camera(22.5, focal_length, (center_points[0][0] - center_points[1][0])))

    if corners[1][0][0] <= corners[2][0][0]:
        tmp = corners[2]
        corners[2] = corners[1]
        corners[1] = tmp


    for i in range(0, 4):
        print("w1", corners[i][1][0] - corners[i][0][0])
        print("w2", corners[i][2][0] - corners[i][3][0])
        print("h1", corners[i][3][1] - corners[i][0][1])
        print("h2", corners[i][2][1] - corners[i][1][1])
        print("top angle:", angle_change(corners[i][3], corners[i][2]))
        print("top reverse:", angle_change(corners[i][2], corners[i][3]))
        print("bottom angle:", angle_change(corners[i][1], corners[i][0]))
        print("bottom reverse:", angle_change(corners[i][0], corners[i][1]))
        print("left angle:", angle_change(corners[i][3], corners[i][0]))
        print("left reverse:", angle_change(corners[i][0], corners[i][3]))
        print("right angle:", angle_change(corners[i][2], corners[i][1]))
        print("right reverse:", angle_change(corners[i][1], corners[i][2]))
    # cv2.circle(threshold_image, (int(cent[0]), int(cent[1])), 1, (100, 100, 100), thickness=7, lineType=8, shift=0)
    # print("leftmost_contour[0][0]", leftmost_contour[0][0])

    return leftmost_contour, center_points, areas, distances, corners


def get_destination_points(corners):
    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)

    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w


# def calculate_homography():
#
#
# def overlay_image(background, overlayImg, hpos, vpos, angle, percentage):
#     # scale down
#     print("percentage", percentage)


def compute_reference(res_ref1, reference_marker_image):
    leftmost_contour, center_point, areas, distances, corners = grab_contour(reference_marker_image)

    homography_points = []
    print("====== Reference Image ======")
    # for i in range(0, 4):
    #     for j in range(0, 4):
    #         cv2.circle(reference_marker_image, tuple(corners[i][j]), 1, (100, 100, 100), thickness=2, lineType=8,
    #                    shift=0)
    #     cv2.circle(reference_marker_image, tuple(corners[0][0]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    # cv2.imshow("ref_corner_test", reference_marker_image)

    cv2.circle(res_ref1, (corners[0][0][0], corners[0][0][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.circle(res_ref1, (corners[1][1][0], corners[1][1][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.circle(res_ref1, (corners[3][2][0], corners[3][2][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.circle(res_ref1, (corners[2][3][0], corners[2][3][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.imshow("Ref marker color", res_ref1)
    # corners
    # for i in range(0,2):
    #     for j in range(0,2):
    #         homography_points.append(corners[i][j])
    # print("homography points:", homography_points)
    # 불필요
    # homography_points.append(corners[0][0])
    # homography_points.append(corners[0][2])
    # homography_points.append(corners[1][0])
    # homography_points.append(corners[1][2])
    # homography_points.append(corners[2][0])
    # homography_points.append(corners[2][2])
    # first_corners = corners[:len(corners) // 2]
    # second_corners = corners[len(corners) // 2:]
    # print("Reference first corners:", first_corners)
    # print("Reference second corners", second_corners)

    # distance of each marker
    # print("distance:", '{:.4}'.format(distances))
    print("Reference distances:", distances)

    # center points
    print("Reference center points:", center_point)
    center_angle_1 = angle_change(center_point[0], center_point[1])
    print("First to Second Line Angle", center_angle_1)
    center_angle_2 = angle_change(center_point[1], center_point[3])
    print("Second to Fourth Line Angle", center_angle_2)
    center_angle_3 = angle_change(center_point[2], center_point[3])
    print("Third to Fourth Line Angle", center_angle_3)
    center_angle_4 = angle_change(center_point[0], center_point[2])
    print("First to Third Line Angle", center_angle_4)

    # area of each marker
    print("Reference area:", areas)

    # distance between markers
    dist_markers = math.sqrt(
        (center_point[0][0] - center_point[1][0]) ** 2 + (center_point[0][1] - center_point[1][1]) ** 2)
    print("Distance between markers in ref markers top", dist_markers)
    dist_markers = math.sqrt(
        (center_point[1][0] - center_point[3][0]) ** 2 + (center_point[1][1] - center_point[3][1]) ** 2)
    print("Distance between markers in ref markers right", dist_markers)
    dist_markers = math.sqrt(
        (center_point[2][0] - center_point[3][0]) ** 2 + (center_point[2][1] - center_point[3][1]) ** 2)
    print("Distance between markers in ref markers bottom", dist_markers)
    dist_markers = math.sqrt(
        (center_point[0][0] - center_point[2][0]) ** 2 + (center_point[0][1] - center_point[2][1]) ** 2)
    print("Distance between markers in ref markers left", dist_markers)
    # for i in range(0, 4):
    #     cv2.circle(reference_marker_image, (center_point[i]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    # cv2.imshow("reference_marker_image", reference_marker_image)
    return leftmost_contour, center_point, corners, distances, homography_points


def compute_moved(res_moved1, moved_marker_image):
    leftmost_contour, center_point, areas, distances, corners = grab_contour(moved_marker_image)
    homography_points, corner_differences = [], []
    print("====== Moved Image ======")
    # print("ref_corners", corners)
    # for i in range(0, 4):
    #     for j in range(0, 4):
    #         cv2.circle(moved_marker_image, tuple(corners[i][j]), 1, (100, 100, 100), thickness=2, lineType=8,
    #                    shift=0)
        # cv2.circle(moved_marker_image, tuple(ref_corners[0][0]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    # cv2.imshow("moved_corner_test", moved_marker_image)
    cv2.circle(res_moved1, (corners[0][0][0], corners[0][0][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.circle(res_moved1, (corners[1][1][0], corners[1][1][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.circle(res_moved1, (corners[3][2][0], corners[3][2][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.circle(res_moved1, (corners[2][3][0], corners[2][3][1]), 1, (255, 140, 65), thickness=5, lineType=8,
               shift=0)
    cv2.imshow("Moved marker color", res_moved1)
    # corners
    # for i in range(0,2):
    #     for j in range(0,2):
    #         homography_points.append(corners[i][j])
    #         print("i", i)
    #         print("j", j)
    # homography_points.append(corners[0][0])
    # homography_points.append(corners[0][2])
    # homography_points.append(corners[1][0])
    # homography_points.append(corners[1][2])
    # homography_points.append(corners[2][0])
    # homography_points.append(corners[2][2])
    # # print("corners", corners)

    # first_corners = corners[:len(corners) // 3]
    # second_corners = corners[len(corners) // 3]
    # third_corners = corners[len(corners) // 3:]
    first_corners = corners[:len(corners) // 2]
    second_corners = corners[len(corners) // 2:]
    # print("Reference first corners:", first_corners)
    # print("Reference first corners:", np.asarray(corners[1][0], dtype=np.float32))
    #
    # print("Reference second corners", second_corners)
    # print("Reference third corners", third_corners)
    # print("first_corners[0][0]", first_corners[0][0])

    # print("second_corners[0][1]", second_corners[0][1])
    # w1 = np.sqrt((second_corners[0][0] - first_corners[0][0]) ** 2 +
    # (second_corners[0][0] - first_corners[0][0]) ** 2)
    # print("w1", w1)
    w1 = np.sqrt((corners[1][0] - corners[0][0]) ** 2 + (corners[1][0] - corners[0][0]) ** 2)
    # print("w1", w1[1])
    corner_differences.append(np.array(first_corners) - np.array(second_corners))

    # print("moved corner differences", corner_differences)
    # distance of each marker

    center_angle_1 = angle_change(center_point[0], center_point[1])
    center_angle_2 = angle_change(center_point[1], center_point[3])
    center_angle_3 = angle_change(center_point[2], center_point[3])
    center_angle_4 = angle_change(center_point[0], center_point[2])

    # area of each marker
    print("Moved area:", areas)

    # distance between markers
    dist_markers = math.sqrt(
        (center_point[0][0] - center_point[1][0]) ** 2 + (center_point[0][1] - center_point[1][1]) ** 2)
    dist_markers = math.sqrt(
        (center_point[1][0] - center_point[3][0]) ** 2 + (center_point[1][1] - center_point[3][1]) ** 2)
    dist_markers = math.sqrt(
        (center_point[2][0] - center_point[3][0]) ** 2 + (center_point[2][1] - center_point[3][1]) ** 2)
    dist_markers = math.sqrt(
        (center_point[0][0] - center_point[2][0]) ** 2 + (center_point[0][1] - center_point[2][1]) ** 2)
    # for i in range(0,4):
    #     cv2.circle(moved_marker_image, (center_point[i]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    # cv2.imshow("Moved_marker_image", moved_marker_image)
    return leftmost_contour, center_point, corners, distances, homography_points


def rot2eul(r):
    beta = -np.arcsin(r[2, 0])
    alpha = np.arctan2(r[2, 1]/np.cos(beta), r[2, 2]/np.cos(beta))
    gamma = np.arctan2(r[1, 0]/np.cos(beta), r[0, 0]/np.cos(beta))

    return np.array((alpha, beta, gamma))


def eul2rot(theta):
    r = np.array([[np.cos(theta[1])*np.cos(theta[2]), np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]), np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]), np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]), np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                 np.sin(theta[0])*np.cos(theta[1]), np.cos(theta[0])*np.cos(theta[1])]])
    return r


def transformPoints(x, y, reverse=False, integer=True):
    if not reverse:
        H = transform_matrix
    else:
        val, H = cv2.invert(transform_matrix)

    # get the elements in the transform matrix
    h0 = H[0, 0]
    h1 = H[0, 1]
    h2 = H[0, 2]
    h3 = H[1, 0]
    h4 = H[1, 1]
    h5 = H[1, 2]
    h6 = H[2, 0]
    h7 = H[2, 1]
    h8 = H[2, 2]

    tx = (h0 * x + h1 * y + h2)
    ty = (h3 * x + h4 * x + h5)
    tz = (h6 * x + h7 * y + h8)

    if integer:
        px = int(tx / tz)
        py = int(ty / tz)
        Z = int(1 / tz)
    else:
        px = tx / tz
        py = ty / tz
        Z = 1 / tz

    return px, py


def unwarp(img, src, dst):
    h, w = img.shape[:2]
    src1 = []
    dst1 = []
    print("src", src)
    print("src[0]", src[0])

    # print("dst:", dst)
    # for i in range(0,5):
    #     cv2.circle(img, tuple(src), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    src1.append(src[0])
    src1.append(src[1])
    src1.append(src[2])
    src1.append(src[3])

    dst1.append(dst[0])
    dst1.append(dst[1])
    dst1.append(dst[2])
    dst1.append(dst[3])
    H, _ = cv2.findHomography(np.asarray(src1, dtype=np.float32), np.asarray(dst1, dtype=np.float32), method=cv2.RANSAC,
                              ransacReprojThreshold=5.0)
    # print('\nThe homography matrix is: \n', H)
    theta = - math.atan2(H[0, 1], H[0, 0]) * 180 / math.pi
    u, _, vh = np.linalg.svd(H[0:2, 0:2])
    R = u @ vh
    angle = math.atan2(R[1, 0], R[0, 0])
    print("theta: ", theta)
    # print("angle: ", angle)
    # p1 = H * src[0]
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    pig = cv2.imread("images/pig.png")
    # M = cv2.getPerspectiveTransform(src, dst)
    # un_warp_pig = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    # print("un_wrap_pig", un_warped)
    # result = cv2.perspectiveTransform(src[None, :, :], M)
    # print("result", result)
    # # cv2.imshow("unwarppig1", un_warp_pig)
    # cv2.imshow("what", un_warped)

    ang = rot2eul(H)
    print("ang", ang)
    hello = eul2rot(ang)
    # print("hello", hello)
    sin_list = np.sin(hello)
    cos_list = np.cos(hello)
    A = np.eye(3)
    c = 0
    for i in range(1, 3):
        for j in range(i):
            ri = np.copy(A[i])
            rj = np.copy(A[j])

            A[i] = cos_list[c] * ri + sin_list[c] * rj
            A[j] = -sin_list[c] * ri + cos_list[c] * rj
            c += 1

    # print("A.T", A.T)
    import transforms3d.euler as eul
    ang = eul.mat2euler(H, axes='sxyz')
    # print("ang2", ang)
    what = eul.euler2mat(ang[0], ang[1], ang[2], axes='sxyz')
    # print("what112", what)
    # overlay_image(img, pig, 230, 250, theta, calcaulate_dist_diff(img))

    # plot
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # # f.subplots_adjust(hspace=.2, wspace=.05)
    # ax1.imshow(img)
    # ax1.set_title('Original Image')
    #
    # x = [src[0][0], src[1][0], src[2][0], src[3][0], src[0][0]]
    # y = [src[0][1], src[1][1], src[2][1], src[3][1], src[0][1]]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img)
    ax1.set_title('Original Image')

    # x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    # y = [src[0][1], src[1][1], src[2][1], src[3][1], src[0][1]]

    # ax2.imshow(img)
    # ax2.plot(x, y, color='yellow', linewidth=3)
    # ax2.set_ylim([h, 0])
    # ax2.set_xlim([0, w])
    # ax2.set_title('Target Area')
    #
    # plt.show()
    return un_warped


def marker_calculation():
    reference_marker_image = cv2.imread('images/1114_1m_transDown_0.jpg')
    moved_marker_image = cv2.imread('images/1114_1m_transDown_6.jpg')
    # overlay_image = cv2.imread('images/pig.png')
    cv2.imshow("Resized ref image", resize_image(reference_marker_image))
    cv2.imshow("Resized moved image", resize_image(moved_marker_image))
    res_ref1, reference_marker_image = apply_threshold(resize_image(reference_marker_image))
    res_moved1, moved_marker_image = apply_threshold(resize_image(moved_marker_image))
    cv2.imshow("res_Ref1", res_ref1)
    # Center of image
    center_X = 189
    # center_X = int(reference_marker_image.shape[1] / 2)
    # center_Y = int(reference_marker_image.shape[0] / 2)
    center_Y = 392
    print("Cetner", center_X, center_Y)
    ref_leftmost, ref_center, ref_corners, ref_distances, ref_homography_points = compute_reference(res_ref1, reference_marker_image)
    moved_leftmost, moved_center, moved_corners, moved_distances, moved_homography_points = compute_moved(res_moved1, moved_marker_image)
    angle_computation = []

    angle_difference, center_difference, corners_difference, distance_difference, homography_corners = [], [], [], [], []
    center_testing, angle_difference_reverse = [], []

    print("====== Differences ======")

    for i in range(0, len(ref_center)):
        # center angle change
        angle_difference.append(angle_change(ref_center[i], moved_center[i]))
        angle_difference_reverse.append(angle_change(moved_center[i], ref_center[i]))
        # center point change
        center_difference.append(math.sqrt(
            (ref_center[i][0] - moved_center[i][0]) ** 2 + (ref_center[i][1] - moved_center[i][1]) ** 2))

        # markers' camera distance difference
        distance_difference.append(ref_distances[i] - moved_distances[i])

        # angle_computation.append(getAngle(ref_center[0], ref_center[1], ref_center[2]))
        # angle_computation.append(getAngle(ref_center[1], ref_center[2], ref_center[0]))
        # angle_computation.append(getAngle(ref_center[2], ref_center[0], ref_center[1]))
        # angle_computation.append(getAngle(moved_center[0], moved_center[1], moved_center[2]))
        # angle_computation.append(getAngle(moved_center[1], moved_center[2], moved_center[0]))
        # angle_computation.append(getAngle(moved_center[2], moved_center[0], moved_center[1]))

        center_testing.append(ref_center[i][0] - moved_center[i][0])
        center_testing.append(ref_center[i][1] - moved_center[i][1])

    # overlay_image(moved_marker_image, overlay_image, moved_center[0][0], moved_center[0][1], angle, percentage)

    first_ref_marker_degrees = []
    first_moved_marker_degrees = []
    second_ref_marker_degrees = []
    second_moved_marker_degrees = []
    moved_adjustments = []
    ref_x_changes, ref_y_changes, moved_x_changes, moved_y_changes = [], [], [], []
    ref_tangents, moved_tangents = [], []
    print("\n")
    ref_x_changes.append(ref_corners[0][0][0] - center_X)
    ref_x_changes.append(ref_corners[1][1][0] - center_X)
    ref_x_changes.append(ref_corners[2][3][0] - center_X)
    ref_x_changes.append(ref_corners[3][2][0] - center_X)
    ref_y_changes.append(ref_corners[0][0][1] - center_Y)
    ref_y_changes.append(ref_corners[1][1][1]- center_Y)
    ref_y_changes.append(ref_corners[2][3][1]- center_Y)
    ref_y_changes.append(ref_corners[3][2][1]- center_Y)
    moved_x_changes.append(moved_corners[0][0][0] - center_X)
    moved_x_changes.append(moved_corners[1][1][0] - center_X)
    moved_x_changes.append(moved_corners[2][3][0] - center_X)
    moved_x_changes.append(moved_corners[3][2][0] - center_X)
    moved_y_changes.append(moved_corners[0][0][1]- center_Y)
    moved_y_changes.append(moved_corners[1][1][1]- center_Y)
    moved_y_changes.append(moved_corners[2][3][1]- center_Y)
    moved_y_changes.append(moved_corners[3][2][1]- center_Y)
    test_angle = 180 - math.degrees(math.atan(abs((moved_corners[0][0][1] - center_Y)) / abs((moved_corners[0][0][0] - center_X))))
    test_angle2 = math.degrees(math.atan(abs((moved_corners[1][1][1] - center_Y)) / abs((moved_corners[1][1][0] - center_X))))

    if ref_x_changes[0] * ref_x_changes[1] < 0 or ref_x_changes[0] * ref_x_changes[1] == 0:
        for i in range(0, 4):
            ref_tangents.append(math.degrees(math.atan(ref_x_changes[i] / ref_y_changes[i])))  # marker's first corner tangent inverse (outer angle)
        for i in range(0, 4):
            ref_tangents.append(math.degrees(math.atan(ref_y_changes[i] / ref_x_changes[i])))
        print("ref_tangents: ", ref_tangents)
        ref_summation_up = ref_tangents[0] + abs(ref_tangents[1])
        ref_summation_down = abs(ref_tangents[2]) + abs(ref_tangents[3])
        ref_summation_left = abs(ref_tangents[4]) + abs(ref_tangents[6])
        ref_summation_right = abs(ref_tangents[5]) + abs(ref_tangents[7])
        # print("ref_summation_up", ref_summation_up)
        # print("ref_summation_down", ref_summation_down)
        # print("ref_summation_left", ref_summation_left)
        # print("ref_summation_right", ref_summation_right)
        # print("\n")

    moved_tangents_test_X, moved_tangents_test_Y = [], []
    moved_tangents_test = []
    adjustments_top_test = []
    print("Dist1: Ref Y coordinate Top", (center_Y - (ref_corners[0][0][1] + ref_corners[1][1][1]) / 2))
    print("Dist2: Ref Y coordinate Bottom", ((ref_corners[2][3][1] + ref_corners[3][2][1]) / 2) - center_Y)
    print("Dist3: Ref X coordinate Left", (center_X - (ref_corners[0][0][0] + ref_corners[2][3][0]) / 2))
    print("Dist4: Ref X coordinate Right", ((ref_corners[1][1][0] + ref_corners[3][2][0]) / 2) - center_X)

    for i in range(0, 4):
        moved_tangents_test_X.append(math.degrees(math.atan(moved_x_changes[i] / moved_y_changes[i])))
        moved_tangents_test.append(math.degrees(math.atan(moved_x_changes[i] / moved_y_changes[i])))
        moved_tangents_test_Y.append(math.degrees(math.atan(moved_y_changes[i] / moved_x_changes[i])))
    print("moved_tangents_test_X: ", moved_tangents_test_X)
    # Y changes
    print("moved_tangents_test_Y: ", moved_tangents_test_Y)
    summation_testing_up_test = abs(moved_tangents_test_X[1] - moved_tangents_test_X[0])
    summation_testing_down_test = abs(moved_tangents_test_X[2] - moved_tangents_test_X[3])
    summation_testing_left_test = abs(moved_tangents_test_Y[0] - moved_tangents_test_Y[2])
    summation_testing_right_test = abs(moved_tangents_test_Y[1] - moved_tangents_test_Y[3])
    # print("summation_testing_up_test", summation_testing_up_test)
    # print("summation_testing_down_test", summation_testing_down_test)
    # print("summation_testing_left_test", summation_testing_left_test)
    # print("summation_testing_right_test", summation_testing_right_test)
    x_ratio = []
    moved_top_left_ratio_test = (ref_tangents[0] / ref_summation_up) * summation_testing_up_test
    x_ratio.append((ref_tangents[0] / ref_summation_up) * summation_testing_up_test)
    moved_top_right_ratio_test = (ref_tangents[1] / ref_summation_up) * summation_testing_up_test
    x_ratio.append((ref_tangents[1] / ref_summation_up) * summation_testing_up_test)
    moved_bottom_left_ratio_test = (ref_tangents[2] / ref_summation_down) * summation_testing_down_test
    x_ratio.append((ref_tangents[2] / ref_summation_down) * summation_testing_down_test)
    moved_bottom_right_ratio_test = (ref_tangents[3] / ref_summation_down) * summation_testing_down_test
    x_ratio.append((ref_tangents[3] / ref_summation_down) * summation_testing_down_test)
    # print("moved_top_left_Ratio_test_", moved_top_left_ratio_test)
    # print("moved_top_right_ratio_test", moved_top_right_ratio_test)
    # print("moved_bottom_left_ratio_test", moved_bottom_left_ratio_test)
    # print("moved_bottom_right_ratio_test", moved_bottom_right_ratio_test)

    # if moved_tangents_test_X[0] > 0:
    adjustments_top_test.append((moved_tangents_test_X[0] - moved_top_left_ratio_test) * -1)
    adjustments_top_test.append(moved_tangents_test_X[1] - moved_top_right_ratio_test)
    adjustments_top_test.append(moved_tangents_test_X[2] - moved_bottom_left_ratio_test)
    adjustments_top_test.append((moved_tangents_test_X[3] - moved_bottom_right_ratio_test) * -1)
    # if moved_tangents_test_X[0] < 0:
    #     adjustments_top_test.append((moved_tangents_test_X[0] - moved_top_left_ratio_test) * -1)
    #     adjustments_top_test.append(moved_tangents_test_X[1] - moved_top_right_ratio_test)
    #     adjustments_top_test.append(moved_tangents_test_X[2] - moved_bottom_left_ratio_test)
    #     adjustments_top_test.append((moved_tangents_test_X[3] - moved_bottom_right_ratio_test) * -1)
    # print("adjustments_top_Test", adjustments_top_test)
    y_ratio = []

    moved_left_top_ratio_test = (ref_tangents[4] / ref_summation_left) * summation_testing_left_test
    y_ratio.append((ref_tangents[4] / ref_summation_left) * summation_testing_left_test)
    moved_right_top_ratio_test = (ref_tangents[5] / ref_summation_right) * summation_testing_right_test
    y_ratio.append((ref_tangents[5] / ref_summation_right) * summation_testing_right_test)
    moved_left_bottom_ratio_test = (ref_tangents[6] / ref_summation_left) * summation_testing_left_test
    y_ratio.append((ref_tangents[6] / ref_summation_left) * summation_testing_left_test)
    moved_right_bottom_ratio_test = (ref_tangents[7] / ref_summation_right) * summation_testing_right_test
    y_ratio.append((ref_tangents[7] / ref_summation_right) * summation_testing_right_test)
    # print("moved_left_top_ratio_test", moved_left_top_ratio_test)
    # print("moved_right_top_ratio_test", moved_right_top_ratio_test)
    # print("moved_bottom_left_ratio_test", moved_left_bottom_ratio_test)
    # print("moved_right_bottom_ratio_test", moved_right_bottom_ratio_test)
    testing_y_index, testing_x_index = [], []
    print("x_ratio", x_ratio)
    print("y_ratio", y_ratio)
    for i in range(0, 4):
        if moved_tangents_test_X[i] * x_ratio[i] < 0:
            testing_x_index.insert(i, moved_tangents_test_X[i] + x_ratio[i])
        if moved_tangents_test_Y[i] * y_ratio[i] < 0:
            testing_y_index.insert(i, moved_tangents_test_Y[i] + y_ratio[i])
        else:
            testing_x_index.insert(i, x_ratio[i] - moved_tangents_test_X[i])
            testing_y_index.insert(i, y_ratio[i] - moved_tangents_test_Y[i])

    testing_x_index[1] = testing_x_index[1] * -1
    testing_x_index[2] = testing_x_index[2] * -1
    testing_y_index[1] = testing_y_index[1] * -1
    testing_y_index[2] = testing_y_index[2] * -1

    print("AVERAGE?", (testing_x_index[0]+ testing_x_index[2])/2)
    print("Gamma: testing_x_index_value", testing_x_index)
    print("Gamma: testing_y_index_value", testing_y_index)
    if moved_tangents_test_Y[1] < 0:
        adjustments_top_test.append(moved_tangents_test_Y[0] - moved_left_top_ratio_test)
        adjustments_top_test.append(moved_tangents_test_Y[1] - moved_right_top_ratio_test)
        adjustments_top_test.append(moved_tangents_test_Y[2] - moved_left_bottom_ratio_test)
        adjustments_top_test.append((moved_tangents_test_Y[3] - moved_right_bottom_ratio_test) * -1)
    if moved_tangents_test_Y[1] > 0:
        adjustments_top_test.append(moved_left_top_ratio_test - moved_tangents_test_Y[0])
        adjustments_top_test.append((moved_tangents_test_Y[1] + moved_right_top_ratio_test) * -1)
        adjustments_top_test.append(moved_tangents_test_Y[2] - moved_left_bottom_ratio_test)
        adjustments_top_test.append(moved_tangents_test_Y[3] + moved_right_bottom_ratio_test)
    # print(">>>>>>adjustments_top_Test", adjustments_top_test)
    # top_line_distance = calculate_distance(ref_corners[0][0][0], ref_corners[0][0][1], )
    m = calculate_distance(moved_corners[0][0][0], moved_corners[0][0][1], center_X, center_Y)
    n = calculate_distance(moved_corners[1][1][0], moved_corners[1][1][1], center_X, center_Y)
    # print("m, n", m, n)
    p = calculate_distance(moved_corners[2][3][0], moved_corners[2][3][1], center_X, center_Y)
    q = calculate_distance(moved_corners[3][2][0], moved_corners[3][2][1], center_X, center_Y)
    # print("moved_top_left_ratio_test", moved_top_left_ratio_test)
    # print("moved_top_right_ratio_test", moved_top_right_ratio_test)
    alpha = math.sin(abs(math.radians(moved_top_left_ratio_test)))
    betha = math.sin(abs(math.radians(moved_top_right_ratio_test)))
    new_x_coordinate = int((m * alpha * moved_corners[1][1][0] + n * betha * moved_corners[0][0][0]) / (m * alpha + n * betha))
    new_y_coordinate = int((m * alpha * moved_corners[1][1][1] + n * betha * moved_corners[0][0][1]) / (m * alpha + n * betha))
    new_x_coordinate1 = int((p * alpha * moved_corners[3][2][0] + q * betha * moved_corners[2][3][0]) / (p * alpha + q * betha))
    new_y_coordinate1 = int(
        (p * alpha * moved_corners[3][2][1] + q * betha * moved_corners[2][3][1]) / (p * alpha + q * betha))
    # print("new_y_coordinate", new_y_coordinate)

    center_distance1 = calculate_distance(new_x_coordinate, new_y_coordinate, center_X, center_Y)
    center_distance2 = calculate_distance(new_x_coordinate1, new_y_coordinate1, center_X, center_Y)

    print("Center_distance_top", center_distance1)
    print("center distance bottom", center_distance2)



    original_first_x_dist = ref_corners[0][0][0] - center_X
    original_second_x_dist = ref_corners[1][1][0] - center_X
    original_first_y_dist = ref_corners[0][0][1] - center_Y
    original_second_y_dist = ref_corners[1][1][1] - center_Y

    moved_first_x_dist = abs(moved_corners[0][0][0]) - abs(new_x_coordinate)
    moved_second_x_dist = abs(moved_corners[1][1][0]) - abs(new_x_coordinate)
    moved_first_y_dist = abs(moved_corners[0][0][1]) - abs(new_y_coordinate)
    moved_second_y_dist = abs(moved_corners[1][1][1]) - abs(new_y_coordinate)

    print("original_x_dist: ", original_first_x_dist, original_second_x_dist)
    print("original_y_dist: ", original_first_y_dist, original_second_y_dist)
    print("moved_x_dist: ", moved_first_x_dist, moved_second_x_dist)
    print("moved_y_dist: ", moved_first_y_dist, moved_second_y_dist)

    # cv2.circle(moved_marker_image, (399, 466), 1, (100, 100, 100), thickness=8, lineType=8, shift=0)
    # cv2.circle(reference_marker_image, (399, 466), 1, (100, 100, 100), thickness=8, lineType=8, shift=0)
    # cv2.imshow("circle", moved_marker_image)
    # cv2.imshow("circle22", reference_marker_image)

    f = calculate_distance(moved_corners[0][0][0], moved_corners[0][0][1], center_X, center_Y)
    g = calculate_distance(moved_corners[2][3][0], moved_corners[2][3][1], center_X, center_Y)

    s = calculate_distance(moved_corners[1][1][0], moved_corners[1][1][1], center_X, center_Y)
    t = calculate_distance(moved_corners[3][2][0], moved_corners[3][2][1], center_X, center_Y)

    alpha2 = math.sin(abs(math.radians(moved_right_top_ratio_test)))
    betha2 = math.sin(abs(math.radians(moved_right_bottom_ratio_test)))

    new_x_coordinate2 = int(
        (f * alpha2 * moved_corners[2][3][0] + g * betha2 * moved_corners[0][0][0]) / (f * alpha2 + g * betha2))
    new_y_coordinate2 = int(
        (f * alpha2 * moved_corners[2][3][1] + g * betha2 * moved_corners[0][0][1]) / (f * alpha2 + g * betha2))

    new_x_coordinate3 = int(
        (s * alpha2 * moved_corners[3][2][0] + t * betha2 * moved_corners[1][1][0]) / (s * alpha2 + t * betha2))
    new_y_coordinate3 = int(
        (s * alpha2 * moved_corners[3][2][1] + t * betha2 * moved_corners[1][1][1]) / (s * alpha2 + t * betha2))

    # M' change
    print("Coordinate change Top: ({}, {})".format(new_x_coordinate, new_y_coordinate))
    print("Coordinate change Bottom: ({}, {})".format(new_x_coordinate1, new_y_coordinate1))
    print("Coordinate change Left: ({}, {})".format(new_x_coordinate2, new_y_coordinate2))
    print("Coordinate change Right: ({}, {})".format(new_x_coordinate3, new_y_coordinate3))
    print("\n")

    # Distance Adjustment
    print("Original Distance TOp:", calculate_distance(center_X, center_Y, ((ref_corners[0][0][0] + ref_corners[1][1][0]) / 2), ((ref_corners[0][0][1] + ref_corners[1][1][1]) / 2)))
    print("MOved distnace top:", calculate_distance(center_X, center_Y, new_x_coordinate, new_y_coordinate))
    print("Original Distance Bottom:",
          calculate_distance(center_X, center_Y, ((ref_corners[2][3][0] + ref_corners[3][2][0]) / 2),
                             ((ref_corners[2][3][1] + ref_corners[3][2][1]) / 2)))
    print("MOved distnace bottom:", calculate_distance(center_X, center_Y, new_x_coordinate1, new_y_coordinate1))
    print("Original Distance Left:",
          calculate_distance(center_X, center_Y, ((ref_corners[0][0][0] + ref_corners[2][3][0]) / 2),
                             ((ref_corners[0][0][1] + ref_corners[2][3][1]) / 2)))
    print("MOved distnace left:", calculate_distance(center_X, center_Y, new_x_coordinate2, new_y_coordinate2))
    print("Original Distance right:",
          calculate_distance(center_X, center_Y, ((ref_corners[1][1][0] + ref_corners[3][2][0]) / 2),
                             ((ref_corners[1][1][1] + ref_corners[3][2][1]) / 2)))
    print("MOved distnace right:", calculate_distance(center_X, center_Y, new_x_coordinate3, new_y_coordinate3))
    print("dist diff top:", calculate_distance(center_X, center_Y, ((ref_corners[0][0][0] + ref_corners[1][1][0]) / 2), ((ref_corners[0][0][1] + ref_corners[1][1][1]) / 2)) - calculate_distance(center_X, center_Y, new_x_coordinate, new_y_coordinate))
    print("dist diff bottom: ", calculate_distance(center_X, center_Y, ((ref_corners[2][3][0] + ref_corners[3][2][0]) / 2),
                             ((ref_corners[2][3][1] + ref_corners[3][2][1]) / 2)) - calculate_distance(center_X, center_Y, new_x_coordinate1, new_y_coordinate1))
    print("dist diff left:", calculate_distance(center_X, center_Y, ((ref_corners[0][0][0] + ref_corners[2][3][0]) / 2),
                             ((ref_corners[0][0][1] + ref_corners[2][3][1]) / 2)) - calculate_distance(center_X, center_Y, new_x_coordinate2, new_y_coordinate2))
    print("dist diff right:", calculate_distance(center_X, center_Y, ((ref_corners[1][1][0] + ref_corners[3][2][0]) / 2),
                             ((ref_corners[1][1][1] + ref_corners[3][2][1]) / 2)) - calculate_distance(center_X, center_Y, new_x_coordinate3, new_y_coordinate3))
    print("\n")

    # X and Y re-location
    print("Coor diff top: ({}, {})".format(center_X - new_x_coordinate, (((moved_corners[0][0][1] + moved_corners[1][1][1]) / 2) - new_y_coordinate)))
    print("Coor diff bottom: ({}, {})".format(center_X - new_x_coordinate1,
                                           (((moved_corners[2][3][1] + moved_corners[3][2][1]) / 2) - new_y_coordinate1)))
    print("Coor diff left: ({}, {})".format((((moved_corners[0][0][0] + moved_corners[2][3][0]) / 2) - new_x_coordinate2), center_Y - new_y_coordinate2))
    print("Coor diff right: ({}, {})".format((((moved_corners[1][1][0] + moved_corners[3][2][0]) / 2) - new_x_coordinate3), center_Y - new_y_coordinate3))


    # original_first_x_dist2 = ref_corners[1][1][0] - center_X
    # original_second_x_dist2 = ref_corners[3][2][0] - center_X
    # original_first_y_dist2 = ref_corners[1][1][1] - center_Y
    # original_second_y_dist2 = ref_corners[3][2][1] - center_Y
    #
    # moved_first_x_dist2 = abs(moved_corners[1][1][0]) - abs(new_x_coordinate2)
    # moved_second_x_dist2 = abs(moved_corners[3][2][0]) - abs(new_x_coordinate2)
    # moved_first_y_dist2 = abs(moved_corners[1][1][1]) - abs(new_y_coordinate2)
    # moved_second_y_dist2 = abs(moved_corners[3][2][1]) - abs(new_y_coordinate2)
    # print("\n")
    # print("original_x_dist2: ", original_first_x_dist2, original_second_x_dist2)
    # print("original_y_dist2: ", original_first_y_dist2, original_second_y_dist2)
    # print("moved_x_dist2: ", moved_first_x_dist2, moved_second_x_dist2)
    # print("moved_y_dist2: ", moved_first_y_dist2, moved_second_y_dist2)
    #
    # print("distance difference x1: ", abs(original_first_x_dist) - abs(moved_first_x_dist))
    # print("distance difference x2: ", abs(original_second_x_dist) - abs(moved_second_x_dist))
    # print("distance difference y1: ", abs(original_first_y_dist) - abs(moved_first_y_dist))
    # print("distance difference y2: ", abs(original_second_y_dist) - abs(moved_second_y_dist))
    #
    # print("distance difference x1-2: ", abs(original_first_x_dist2) - abs(moved_first_x_dist2))
    # print("distance difference x2-2: ", abs(original_second_x_dist2) - abs(moved_second_x_dist2))
    # print("distance difference y1-2: ", abs(original_first_y_dist2) - abs(moved_first_y_dist2))
    # print("distance difference y2-2: ", abs(original_second_y_dist2) - abs(moved_second_y_dist2))
    #
    # print("corners:", ref_corners[0][0], ref_corners[1][1], ref_corners[2][3], ref_corners[3][2])
    print("distance X", ref_corners[0][0][0] - center_X, ref_corners[1][1][0] - center_X)
    print("distance y", ref_corners[0][0][1] - center_Y, ref_corners[1][1][1] - center_Y)
    print("distance X2", ref_corners[2][3][0] - center_X, ref_corners[3][2][0] - center_X)
    print("distance y2", ref_corners[2][3][1] - center_Y, ref_corners[3][2][1] - center_Y)


    # IMAGE RESULTS
    # # >> Reference Image_X
    # cv2.line(res_ref1, (center_X, 0), (center_X, res_ref1.shape[0]), (100, 100, 100), thickness=1)
    # cv2.line(res_ref1, (0, center_Y), (res_ref1.shape[1], center_Y), (100, 100, 100), thickness=1)
    # cv2.circle(res_ref1, (center_X, center_Y), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[0][0][0], ref_corners[0][0][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[1][1][0], ref_corners[1][1][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[2][3][0], ref_corners[2][3][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[3][2][0], ref_corners[3][2][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.putText(res_ref1, "alphaT1_x: {:.2f}".format(ref_tangents[0]), (30, center_Y-250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_ref1, "alphaT2_x: {:.2f}".format(abs(ref_tangents[1])), (res_ref1.shape[1] - 160, center_Y-250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_ref1, "alphaB1_x: {:.2f}".format(abs(ref_tangents[2])), (30, center_Y + 250), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_ref1, "alphaB2_x: {:.2f}".format(abs(ref_tangents[3])), (res_ref1.shape[1] - 160, center_Y + 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.circle(res_ref1, (189, 274), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.circle(res_ref1, (189, 511), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.circle(res_ref1, (104, 392), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.circle(res_ref1, (274, 392), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.putText(res_ref1, "d1=119", (155, center_Y - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.putText(res_ref1, "d2=119", (155, center_Y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.putText(res_ref1, "d3=85", (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.putText(res_ref1, "d4=85", (250, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.imshow("Ref tangents", res_ref1)
    # print("ref corners", ref_corners)

    # # >> Reference Image_Y
    # cv2.line(res_ref1, (center_X, 0), (center_X, res_ref1.shape[0]), (100, 100, 100), thickness=1)
    # cv2.line(res_ref1, (0, center_Y), (res_ref1.shape[1], center_Y), (100, 100, 100), thickness=1)
    # cv2.circle(res_ref1, (center_X, center_Y), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[0][0][0], ref_corners[0][0][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[1][1][0], ref_corners[1][1][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[2][3][0], ref_corners[2][3][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_ref1, (center_X, center_Y), (ref_corners[3][2][0], ref_corners[3][2][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.putText(res_ref1, "alphaT1_y: {:.2f}".format(ref_tangents[4]), (30, center_Y - 250), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 255, 255), 2)
    # cv2.putText(res_ref1, "alphaT2_y: {:.2f}".format(abs(ref_tangents[5])), (res_ref1.shape[1] - 160, center_Y - 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_ref1, "alphaB1_y: {:.2f}".format(abs(ref_tangents[6])), (30, center_Y + 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_ref1, "alphaB2_y: {:.2f}".format(abs(ref_tangents[7])), (res_ref1.shape[1] - 160, center_Y + 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.circle(res_ref1, (189, 274), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.circle(res_ref1, (189, 511), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.circle(res_ref1, (104, 392), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.circle(res_ref1, (274, 392), 1, (255, 255, 255), thickness=5, lineType=8, shift=0)
    # cv2.putText(res_ref1, "d1=119", (155, center_Y - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.putText(res_ref1, "d2=119", (155, center_Y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.putText(res_ref1, "d3=85", (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.putText(res_ref1, "d4=85", (250, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.imshow("Ref tangents", res_ref1)
    # print("ref corners", ref_corners)

    # # Top and bottom
    # cv2.line(res_moved1, (center_X, 0), (center_X, res_ref1.shape[0]), (100, 100, 100), thickness=1)
    # cv2.line(res_moved1, (0, center_Y), (res_ref1.shape[1], center_Y), (100, 100, 100), thickness=1)
    # cv2.circle(res_moved1, (new_x_coordinate, new_y_coordinate), 1, (255, 140, 65), thickness=5, lineType=8,
    #            shift=0)
    # cv2.circle(res_moved1, (new_x_coordinate1, new_y_coordinate1), 1, (255, 140, 65), thickness=5, lineType=8,
    #            shift=0)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[0][0][0], moved_corners[0][0][1]), (255,140,60), thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[1][1][0], moved_corners[1][1][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[3][2][0], moved_corners[3][2][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[2][3][0], moved_corners[2][3][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate, new_y_coordinate), (255,255,255), thickness=2)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate1, new_y_coordinate1), (255, 255, 255), thickness=2)
    # cv2.circle(res_moved1, (center_X, center_Y), 1, (255, 255, 255), thickness=10, lineType=8, shift=0)
    # cv2.putText(res_moved1, "M'T", (new_x_coordinate + 10, new_y_coordinate + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # # cv2.putText(res_moved1, "({}, {})".format(new_x_coordinate, new_y_coordinate), (new_x_coordinate - 20, new_y_coordinate - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    # #             (255, 255, 255), 1)
    # # cv2.putText(res_moved1, "({}, {})".format(new_x_coordinate1, new_y_coordinate1),
    # #             (new_x_coordinate1 - 10, new_y_coordinate1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    # #             (255, 255, 255), 1)
    # cv2.putText(res_moved1, "M'B", (new_x_coordinate1 + 10, new_y_coordinate1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "R", (center_X +10, center_Y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaT1': {:.2f}".format(moved_tangents_test_X[0]), (50, center_Y - 250), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaT2': {:.2f}".format(abs(moved_tangents_test_X[1])), (res_moved1.shape[1] - 130, center_Y - 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaB1': {:.2f}".format(abs(moved_tangents_test_X[2])), (50, center_Y + 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaB2': {:.2f}".format(abs(moved_tangents_test_X[3])), (res_moved1.shape[1] - 130, center_Y + 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "Gamma x': {:.2f}".format(abs(testing_x_index[0]+testing_x_index[2])/2),
    #             (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.destroyAllWindows()
    # cv2.imshow("Angle change top/bottom", res_moved1)

    # # Left and Right Angle
    # cv2.line(res_moved1, (center_X, 0), (center_X, res_ref1.shape[0]), (100, 100, 100), thickness=1)
    # cv2.line(res_moved1, (0, center_Y), (res_ref1.shape[1], center_Y), (100, 100, 100), thickness=1)
    # cv2.circle(res_moved1, (new_x_coordinate2, new_y_coordinate2), 1, (255, 255, 255), thickness=5, lineType=8,
    #            shift=0)
    # cv2.circle(res_moved1, (new_x_coordinate3, new_y_coordinate3), 1, (255, 255, 255), thickness=5, lineType=8,
    #            shift=0)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[0][0][0], moved_corners[0][0][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[1][1][0], moved_corners[1][1][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[3][2][0], moved_corners[3][2][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[2][3][0], moved_corners[2][3][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate2, new_y_coordinate2), (255, 255, 255), thickness=2)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate3, new_y_coordinate3), (255, 255, 255), thickness=2)
    # cv2.circle(res_moved1, (center_X, center_Y), 1, (255, 255, 255), thickness=10, lineType=8, shift=0)
    # cv2.putText(res_moved1, "M'T", (new_x_coordinate2 - 20, new_y_coordinate2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # # cv2.putText(res_moved1, "({}, {})".format(new_x_coordinate, new_y_coordinate), (new_x_coordinate - 20, new_y_coordinate - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    # #             (255, 255, 255), 1)
    # # cv2.putText(res_moved1, "({}, {})".format(new_x_coordinate1, new_y_coordinate1),
    # #             (new_x_coordinate1 - 10, new_y_coordinate1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    # #             (255, 255, 255), 1)
    # cv2.putText(res_moved1, "M'B", (new_x_coordinate3 - 20, new_y_coordinate3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "R", (center_X + 10, center_Y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaT1': {:.2f}".format(moved_tangents_test_Y[0]), (50, center_Y - 250),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaT2': {:.2f}".format(abs(moved_tangents_test_Y[1])),
    #             (res_moved1.shape[1] - 130, center_Y - 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaB1': {:.2f}".format(abs(moved_tangents_test_Y[2])), (50, center_Y + 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "alphaB2': {:.2f}".format(abs(moved_tangents_test_Y[3])),
    #             (res_moved1.shape[1] - 130, center_Y + 250),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(res_moved1, "Gamma y': {:.2f}".format(abs(testing_y_index[0] + testing_y_index[1]) / 2),
    #             (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.destroyAllWindows()
    # cv2.imshow("Angle change top/bottom", res_moved1)

    # # Distance change
    # cv2.line(res_moved1, (center_X, 0), (center_X, res_ref1.shape[0]), (100, 100, 100), thickness=1)
    # cv2.line(res_moved1, (0, center_Y), (res_ref1.shape[1], center_Y), (100, 100, 100), thickness=1)
    # cv2.circle(res_moved1, (new_x_coordinate, new_y_coordinate), 1, (255, 140, 65), thickness=5, lineType=8,
    #            shift=0)
    # cv2.circle(res_moved1, (new_x_coordinate1, new_y_coordinate1), 1, (255, 140, 65), thickness=5, lineType=8,
    #            shift=0)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[0][0][0], moved_corners[0][0][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[1][1][0], moved_corners[1][1][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[3][2][0], moved_corners[3][2][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[2][3][0], moved_corners[2][3][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate, new_y_coordinate), (255, 255, 255), thickness=2)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate1, new_y_coordinate1), (255, 255, 255), thickness=2)
    # cv2.circle(res_moved1, (center_X, center_Y), 1, (255, 255, 255), thickness=10, lineType=8, shift=0)
    # cv2.putText(res_moved1, "M'T", (new_x_coordinate + 10, new_y_coordinate + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "{:.2f}".format((center_Y - ref_corners[0][0][1]) - center_distance1),
    #             (new_x_coordinate - 20, new_y_coordinate - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 1)
    # cv2.putText(res_moved1, "{:.2f}".format(ref_corners[3][2][1] - center_Y - center_distance2),
    #             (new_x_coordinate1 - 10, new_y_coordinate1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 1)
    # cv2.putText(res_moved1, "M'B", (new_x_coordinate1 + 10, new_y_coordinate1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "R", (center_X + 10, center_Y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.imshow("Distance change", res_moved1.copy())

    # # >> Location change_X
    # cv2.putText(res_moved1, "X coordinate change", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.line(res_moved1, (center_X, 0), (center_X, res_ref1.shape[0]), (100, 100, 100), thickness=1)
    # cv2.line(res_moved1, (0, center_Y), (res_ref1.shape[1], center_Y), (100, 100, 100), thickness=1)
    # cv2.circle(res_moved1, (new_x_coordinate, new_y_coordinate), 1, (255, 255, 255), thickness=5, lineType=8,
    #            shift=0)
    # cv2.circle(res_moved1, (new_x_coordinate1, new_y_coordinate1), 1, (255, 255, 255), thickness=5, lineType=8,
    #            shift=0)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[0][0][0], moved_corners[0][0][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[1][1][0], moved_corners[1][1][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[3][2][0], moved_corners[3][2][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (moved_corners[2][3][0], moved_corners[2][3][1]), (255, 140, 60),
    #          thickness=1)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate, new_y_coordinate), (255, 255, 255), thickness=2)
    # cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate1, new_y_coordinate1), (255, 255, 255), thickness=2)
    # cv2.circle(res_moved1, (center_X, center_Y), 1, (255, 255, 255), thickness=10, lineType=8, shift=0)
    # cv2.putText(res_moved1, "M'T", (new_x_coordinate + 10, new_y_coordinate + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "x top: {:.2f}".format(int((ref_corners[0][0][0] + ref_corners[1][1][0]) / 2) - new_x_coordinate),
    #             (new_x_coordinate - 45, new_y_coordinate - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "y top: {:.2f}".format(int((ref_corners[0][0][1] + ref_corners[1][1][1]) / 2) - new_y_coordinate),
    #             (new_x_coordinate - 45, new_y_coordinate - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 1)
    # cv2.putText(res_moved1, "x bottom: {:.2f}".format(int((ref_corners[2][3][0] + ref_corners[3][2][0]) / 2) - new_x_coordinate1),
    #             (new_x_coordinate1 - 45, new_y_coordinate1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "y bottom: {:.2f}".format(int((ref_corners[2][3][1] + ref_corners[3][2][1]) / 2) - new_y_coordinate1),
    #             (new_x_coordinate1 - 45, new_y_coordinate1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 1)
    # cv2.putText(res_moved1, "M'B", (new_x_coordinate1 + 10, new_y_coordinate1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.putText(res_moved1, "R", (center_X + 10, center_Y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.imshow("Distance change X", res_moved1.copy())

    # >> Location change_Y
    cv2.putText(res_moved1, "Y coordinate change", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.line(res_moved1, (center_X, 0), (center_X, res_ref1.shape[0]), (100, 100, 100), thickness=1)
    cv2.line(res_moved1, (0, center_Y), (res_ref1.shape[1], center_Y), (100, 100, 100), thickness=1)
    cv2.circle(res_moved1, (new_x_coordinate2, new_y_coordinate2), 1, (255, 255, 255), thickness=5, lineType=8,
               shift=0)
    cv2.circle(res_moved1, (new_x_coordinate3, new_y_coordinate3), 1, (255, 255, 255), thickness=5, lineType=8,
               shift=0)
    cv2.line(res_moved1, (center_X, center_Y), (moved_corners[0][0][0], moved_corners[0][0][1]), (255, 140, 60),
             thickness=1)
    cv2.line(res_moved1, (center_X, center_Y), (moved_corners[1][1][0], moved_corners[1][1][1]), (255, 140, 60),
             thickness=1)
    cv2.line(res_moved1, (center_X, center_Y), (moved_corners[3][2][0], moved_corners[3][2][1]), (255, 140, 60),
             thickness=1)
    cv2.line(res_moved1, (center_X, center_Y), (moved_corners[2][3][0], moved_corners[2][3][1]), (255, 140, 60),
             thickness=1)
    cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate2, new_y_coordinate2), (255, 255, 255), thickness=2)
    cv2.line(res_moved1, (center_X, center_Y), (new_x_coordinate3, new_y_coordinate3), (255, 255, 255), thickness=2)
    cv2.circle(res_moved1, (center_X, center_Y), 1, (255, 255, 255), thickness=10, lineType=8, shift=0)
    cv2.putText(res_moved1, "M'T", (new_x_coordinate2 - 20, new_y_coordinate2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)
    cv2.putText(res_moved1, "x top: {:.2f}".format(int((ref_corners[0][0][0] + ref_corners[2][3][0]) / 2) - new_x_coordinate2),
                (new_x_coordinate - 45, new_y_coordinate - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(res_moved1, "y top: {:.2f}".format(int((ref_corners[0][0][1] + ref_corners[2][3][1]) / 2) - new_y_coordinate2),
                (new_x_coordinate - 45, new_y_coordinate - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)
    cv2.putText(res_moved1, "x bottom: {:.2f}".format(int((ref_corners[1][1][0] + ref_corners[3][2][0]) / 2) - new_x_coordinate3),
                (new_x_coordinate1 - 45, new_y_coordinate1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(res_moved1, "y bottom: {:.2f}".format(int((ref_corners[1][1][1] + ref_corners[3][2][1]) / 2) - new_y_coordinate3),
                (new_x_coordinate1 - 45, new_y_coordinate1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)
    cv2.putText(res_moved1, "M'B", (new_x_coordinate3 - 20, new_y_coordinate3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)
    cv2.putText(res_moved1, "R", (center_X + 10, center_Y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Distance change Y", res_moved1.copy())

    print("ref points", ref_corners)
    print("moved points", moved_corners)

    # if moved_tangents_test[0] * moved_tangents_test[1] < 0 or moved_tangents_test[0] * moved_tangents_test[1] == 0:
    #     print("moved_tangents[0] * moved_tangents[1] < 0 or moved_tangents[0] * moved_tangents[1] == 0:")
    #     summation_testing_up = abs(moved_tangents_test[0]) + abs(moved_tangents_test[1])
    #     summation_testing_down = abs(moved_tangents_test[2]) + abs(moved_tangents_test[3])
    #
    # if moved_tangents_test[0] > 0:
    #     print("if moved_tangents[0] > 0:")
    #     summation_testing_up = moved_tangents_test[0] -moved_tangents_test[1]
    #     summation_testing_down = moved_tangents_test[2] -moved_tangents_test[3]
    #
    # if moved_tangents_test[0] < 0:
    #     print("moved_tangents[0] < 0:")
    #     summation_testing_up = (abs(moved_tangents_test[1]) - abs(moved_tangents_test[0])) * -1
    #     summation_testing_down = (abs(moved_tangents_test[3]) - abs(moved_tangents_test[2])) * -1
    # print("summation_testing_up", summation_testing_up)
    # print("summation_testing_down", summation_testing_down)


    # for i in range(0, 4):
    #     moved_tangents.append(abs(math.degrees(math.atan(moved_x_changes[i] / moved_y_changes[i]))))
    # print("moved_tangents: ", moved_tangents)
    #
    # if moved_x_changes[0] * moved_x_changes[1] < 0 or moved_x_changes[0] * moved_x_changes[1] == 0:
    #     print("MOVED FISRT")
    #     for i in range(0, 4):
    #         moved_tangents.append(abs(math.degrees(math.atan(moved_x_changes[i] / moved_y_changes[i]))))
    #     print("moved_tangents: ", moved_tangents)
    #     moved_summation_up = moved_tangents[0] + abs(moved_tangents[1])
    #     moved_summation_down = abs(moved_tangents[2]) + moved_tangents[3]
    #     print("moved_summation_up", moved_summation_up)
    #     print("moved_summation_down", moved_summation_down)
    #     moved_top_left_ratio = abs((ref_tangents[0] / ref_summation_up) * moved_summation_up)  # + + +
    #     moved_top_right_ratio = abs((ref_tangents[1] / ref_summation_up) * moved_summation_up)  # - + -
    #     moved_down_left_ratio = abs((ref_tangents[2] / ref_summation_down) * moved_summation_down) # - + -
    #     moved_down_right_ratio = abs((ref_tangents[3] / ref_summation_down) * moved_summation_down) # + + -
    #
    #     print("moved_top_left_ratio", moved_top_left_ratio)
    #     print("moved_top_right_ratio", moved_top_right_ratio)
    #     print("moved_down_left_ratio", moved_down_left_ratio)
    #     print("moved_down_right_ratio", moved_down_right_ratio)
    #
    #     moved_adjustments.append(moved_top_left_ratio - moved_tangents[0])
    #     moved_adjustments.append(moved_top_right_ratio - moved_tangents[1])
    #     moved_adjustments.append(moved_down_left_ratio - moved_tangents[2])
    #     moved_adjustments.append(moved_down_right_ratio - moved_tangents[3])
    #
    #     print("moved_adjustments: ", moved_adjustments)
    #
    # # placed on the same side
    # if moved_x_changes[0] * moved_x_changes[1] > 0:
    #     print("MOVED SECOND")
    #     for i in range(0, 4):
    #         moved_tangents.append(abs(math.degrees(math.atan(moved_x_changes[i] / moved_y_changes[i]))))
    #     print("moved_tangents: ", moved_tangents)
    #     moved_summation_up = abs(moved_tangents[1]) - abs(moved_tangents[0])
    #     moved_summation_down = abs(moved_tangents[3]) - abs(moved_tangents[2])
    #     print("moved_summation_up", moved_summation_up)
    #     print("moved_summation_down", moved_summation_down)
    #     moved_top_left_ratio = abs((ref_tangents[0] / ref_summation_up) * moved_summation_up) # + + -
    #     moved_top_right_ratio = abs((ref_tangents[1] / ref_summation_up) * moved_summation_up) # - + -
    #     moved_down_left_ratio = abs((ref_tangents[2] / ref_summation_down) * moved_summation_down) # - + -
    #     moved_down_right_ratio = abs((ref_tangents[3] / ref_summation_down) * moved_summation_down) # + + -
    #
    #     if moved_x_changes[0] < 0:
    #         moved_adjustments.append(moved_top_left_ratio - moved_tangents[0])
    #         moved_adjustments.append(moved_top_right_ratio + moved_tangents[1])
    #         moved_adjustments.append(moved_down_left_ratio - moved_tangents[2])
    #         moved_adjustments.append(moved_down_right_ratio + moved_tangents[3])
    #     if moved_x_changes[0] > 0:
    #         moved_adjustments.append(moved_top_left_ratio + moved_tangents[0])
    #         moved_adjustments.append(moved_top_right_ratio - moved_tangents[1])
    #         moved_adjustments.append(moved_down_left_ratio + moved_tangents[2])
    #         moved_adjustments.append(moved_down_right_ratio - moved_tangents[3])
    #
    #     print("moved_top_left_ratio", moved_top_left_ratio)
    #     print("moved_top_right_ratio", moved_top_right_ratio)
    #     print("moved_down_left_ratio", moved_down_left_ratio)
    #     print("moved_down_right_ratio", moved_down_right_ratio)
    #
    #     print("moved_adjustments: ", moved_adjustments)
    # print("moved_y_changes[0]", moved_y_changes[0])
    # print("moved_y_changes[2]", moved_y_changes[2])
    #
    # # TRANS UP / DOWN
    # if moved_y_changes[0] * moved_y_changes[2] < 0 or moved_y_changes[0] * moved_y_changes[2] == 0:
    #     print("\nMOVED THIRD")
    #     for i in range(0, 4):
    #         moved_tangents.append(abs(math.degrees(math.atan(moved_y_changes[i] / moved_x_changes[i]))))
    #     print("moved_tangents: ", moved_tangents)
    #     moved_summation_left = abs(moved_tangents[8]) + abs(moved_tangents[10])
    #     moved_summation_right = abs(moved_tangents[9]) + abs(moved_tangents[11])
    #     print("moved_summation_left", moved_summation_left)
    #     print("moved_summation_right", moved_summation_right)
    #     moved_left_top_ratio = abs((ref_tangents[4] / ref_summation_left) * moved_summation_left)  # + + +
    #     moved_left_bottom_ratio = abs((ref_tangents[5] / ref_summation_right) * moved_summation_right) # - + -
    #     moved_right_top_ratio = abs((ref_tangents[6] / ref_summation_left) * moved_summation_left)  # - + -
    #     moved_right_bottom_ratio = abs((ref_tangents[7] / ref_summation_right) * moved_summation_right) # + + -
    #
    #     print("moved_left_left_ratio", moved_left_top_ratio)
    #     print("moved_right_top_ratio", moved_right_top_ratio)
    #     print("moved_left_bottom_ratio", moved_left_bottom_ratio)
    #     print("moved_right_bottom_ratio", moved_right_bottom_ratio)
    #
    #     moved_adjustments.append(moved_left_top_ratio - moved_tangents[8])
    #     moved_adjustments.append(moved_left_bottom_ratio - moved_tangents[9])
    #     moved_adjustments.append(moved_right_top_ratio - moved_tangents[10])
    #     moved_adjustments.append(moved_right_bottom_ratio - moved_tangents[11])
    #
    #     print("moved_adjustments: ", moved_adjustments)
    #
    # # Trans up & down - same plane
    # if moved_y_changes[0] * moved_y_changes[2] > 0:
    #     print("\nMOVED Fourth")
    #     for i in range(0, 4):
    #         moved_tangents.append(abs(math.degrees(math.atan(moved_y_changes[i] / moved_x_changes[i]))))
    #     print("moved_tangents: ", moved_tangents)
    #     moved_summation_left = abs(moved_tangents[4]) - abs(moved_tangents[6])
    #     moved_summation_right = abs(moved_tangents[5]) - abs(moved_tangents[7])
    #     print("moved_summation_left", moved_summation_left)
    #     print("moved_summation_right", moved_summation_right)
    #     moved_left_top_ratio = abs((ref_tangents[4] / ref_summation_left) * moved_summation_left)  # + + +
    #     moved_left_bottom_ratio = abs((ref_tangents[6] / ref_summation_left) * moved_summation_left)  # - + -
    #     moved_right_top_ratio = abs((ref_tangents[5] / ref_summation_right) * moved_summation_right)  # - + -
    #     moved_right_bottom_ratio = abs((ref_tangents[7] / ref_summation_right) * moved_summation_right)  # + + -
    #     print("moved+y+changes[2]", moved_y_changes[2])
    #     if moved_y_changes[2] > 0:
    #         moved_adjustments.append(moved_left_top_ratio - moved_tangents[8])
    #         moved_adjustments.append(moved_left_bottom_ratio + moved_tangents[10])
    #         moved_adjustments.append(moved_right_top_ratio - moved_tangents[9])
    #         moved_adjustments.append(moved_right_bottom_ratio + moved_tangents[11])
    #     if moved_y_changes[2] < 0:
    #         moved_adjustments.append(moved_left_top_ratio - moved_tangents[8])
    #         moved_adjustments.append(moved_left_bottom_ratio + moved_tangents[10])
    #         moved_adjustments.append(moved_right_top_ratio - moved_tangents[9])
    #         moved_adjustments.append(moved_right_bottom_ratio + moved_tangents[11])
    #
    #     print("moved_left_top_ratio", moved_left_top_ratio)
    #     print("moved_left_bottom_ratio", moved_left_bottom_ratio)
    #     print("moved_right_top_ratio", moved_right_top_ratio)
    #     print("moved_right_bottom_ratio", moved_right_bottom_ratio)
    #
    #     print("moved_adjustments: ", moved_adjustments)

    # print("reference corners: ", ref_corners)
    # print("moved corners: ", moved_corners)


if __name__ == '__main__':
    marker_calculation()
    cv2.waitKey()
    cv2.destroyAllWindows
