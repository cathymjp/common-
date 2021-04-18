import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
from scipy.spatial import distance as dist
from imutils import contours
from imutils import perspective
from math import atan2,degrees


def resize_image(image):
    # resize image
    scale_percent = 20  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return image


def apply_threshold(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # green_lower = np.array([21, 80, 100], np.uint8)
    # green_upper = np.array([255, 255, 255], np.uint8)

    green_lower = np.array([20, 55, 0], np.uint8)
    green_upper = np.array([255, 255, 255], np.uint8)

    green = cv2.inRange(hsv, green_lower, green_upper)
    image_res = cv2.bitwise_and(image, image, mask=green)
    image_res = cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB)

    image_res_thre = cv2.cvtColor(image_res, cv2.COLOR_RGB2GRAY)
    _, image_res_thre = cv2.threshold(image_res_thre, 255, 255, cv2.THRESH_OTSU)
    # cv2.imshow("thre", image_res_thre)
    return image_res_thre


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


def distance_to_camera(known_width, focal_length, pic_width):
    return (known_width * focal_length) / pic_width


def angle_change(initial, moved):
    x_change = moved[0] - initial[0]
    y_change = moved[1] - initial[1]
    return degrees(atan2(y_change, x_change))


def order_points_old(pts):
    print("pts", pts)
    ptsarray = [pts]
    print("ptssaary", ptsarray)
    # first_corners = pts[:len(pts) // 2]
    rect = np.zeros((4, 2), dtype="float32")
    # top-left point will have smallest sum, whereas bottom-right point will have largest sum
    # s = pts.sum(axis=1)
    # print("s", s)
    # rect[0] = pts[np.argmin(s)]
    # print("rect[0]", rect[0])
    # top = tuple(pts[pts[:, :, 1].argmin()][0])

    rect[0] = tuple(pts[pts[:, :, 0].argmin()][0])

    rect[2] = tuple(pts[pts[:, :, 0].argmax()][0])
    rect[1] = tuple(pts[pts[:, :, 1].argmin()][0])
    rect[3] = tuple(pts[pts[:, :, 1].argmax()][0])
    print("top-left", rect[0])
    print("top-right", rect[1])
    print("bottom-right", rect[2])
    print("bottom-left", rect[3])
    # cv2.circle(image, tuple(rect[0]), 1, (100, 100, 100), thickness=2, lineType=8, shift=0)
    # cv2.circle(image, tuple(rect[1]), 1, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #
    # cv2.circle(image, tuple(rect[2]), 1, (255, 255, 0), thickness=2, lineType=8, shift=0)
    # cv2.circle(image, tuple(rect[3]), 1, (255, 0, 255), thickness=2, lineType=8, shift=0)
    # cv2.imshow("circles", image)
    # print("np.argmax(s)", pts[np.argmax(s)])
    # rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    # diff = np.diff(pts, axis=1)
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def order_points(pts):
    print("pts`````", pts)
    # top-left, top-right, bottom-right, and bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    print("s'''''", s)

    rect[0] = pts[np.argmin(s)]  # 0 original
    rect[2] = pts[np.argmax(s)]  # 2 original

    # compute the difference between points, the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    print("diff", diff)
    rect[1] = pts[np.argmin(diff)]  # 1 original
    rect[3] = pts[np.argmax(diff)]  # 3 original
    print("np.argmin(s)", pts[np.argmin(s[0])])
    print("rect[0]", rect[0])
    print("np.argmax(s)", pts[np.argmax(s)])
    print("Rrrrrrrr", rect)
    return rect


# def corners_from_contour(image, cnt):
#     epsilon = 0.02 * cv2.arcLength(cnt, True)
#     approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
#
#     pt1 = np.array([approx_corners])
#     approx_corners2 = sorted(np.concatenate(pt1).tolist())
#
#     newlist = []
#
#     for item in approx_corners2:
#         for items in item:
#             # print("items:", items)
#             if items not in newlist:
#                 newlist.append(items)
#     # print("newlist:", newlist)
#     # print("newlist type:", type(np.asarray(newlist)))
#     # ordered = order_points(np.asarray(newlist))
#     # ordered2 = np.asarray(np.asarray(newlist))
#     print("ordered2", newlist)
#     # for points in ordered:
#     #     cv2.circle(image, tuple(points), 3, (255, 100, 100), thickness=3, lineType=8, shift=0)
#
#     approx_corners = sorted(np.concatenate(approx_corners).tolist())
#     print("approx_corners", approx_corners)
#     # Rearranging the order of the corner points
#     # approx_corners = [approx_corners[i] for i in [0, 2, 3, 1]]
#
#     return approx_corners


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

# def swap_order(pts):


def grab_contour(threshold_image):
    # sort the contours from left-to-right and initialize the 'pixels per metric' calibration variable
    edge = cv2.Canny(threshold_image, 75, 200)
    cnts = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    new_swap_list = []
    # sort the contours from left-to-right and initialize the 'pixels per petric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    leftmost_contour = None

    center_points, areas, distances, corners, three_areas = [], [], [], [], []
    testing = []
    testing2 = []
    # three_contours = []
    known_width = 7.6
    focal_length = 295
    # print("<<< Grab Contours >>>")
    # cv2.circle(threshold_image, (277,375), 1, (100, 100, 100), thickness=7, lineType=8, shift=0)
    # cv2.imshow("leftmost circle 277,375", threshold_image)
    # cv2.circle(threshold_image, (276,501), 1, (100, 100, 100), thickness=7, lineType=8, shift=0)
    # cv2.imshow("leftmost circle 276,501", threshold_image)
    # cv2.circle(threshold_image, (157,375), 1, (100, 100, 100), thickness=7, lineType=8, shift=0)
    # cv2.imshow("leftmost circle 157,375", threshold_image)
    # cv2.circle(threshold_image, (156,501), 1, (100, 100, 100), thickness=7, lineType=8, shift=0)
    # cv2.imshow("leftmost circle 156,501", threshold_image)
    # get four largest coutour areas
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        three_areas.append(area)
        sorteddata = sorted(zip(three_areas, cnts), key=lambda x: x[0], reverse=True)
    tessss = []
    # Four largest contours' coordinates
    compare_list = [sorteddata[0][1][0][0], sorteddata[1][1][0][0], sorteddata[2][1][0][0], sorteddata[3][1][0][0]]
    first, second, third, fourth = compare_lists([compare_list])
    tessss.append(first)
    tessss.append(second)
    tessss.append(third)
    tessss.append(fourth)

    # print(">?>?", np.argsort(tessss))
    for i in np.argsort(tessss):
        new_swap_list.append(sorteddata[i][1])
    # print("123123", new_swap_list)
    # if first == 0:
    #     new_swap_list.append(sorteddata[0][1])
    #     if second == 1:
    #         new_swap_list.append(sorteddata[1][1])
    #         if third == 2:
    #             new_swap_list.append(sorteddata[2][1])
    #             new_swap_list.append(sorteddata[3][1])
    #         else:
    #             new_swap_list.append(sorteddata[3][1])
    #             new_swap_list.append(sorteddata[2][1])
    #     elif second == 2:
    #         new_swap_list.append(sorteddata[2][1])
    #         new_swap_list.append(sorteddata[1][1])
    #     elif second == 3:
    #         if third == 2:
    #             new_swap_list.append(sorteddata[1][1])
    #             new_swap_list.append(sorteddata[3][1])
    #             new_swap_list.append(sorteddata[2][1])
    # elif second == 0:
    #     new_swap_list.append(sorteddata[1][1])
    #     if first == 1:
    #         new_swap_list.append(sorteddata[0][1])
    #         new_swap_list.append(sorteddata[2][1])
    #     elif first == 2:
    #         new_swap_list.append(sorteddata[2][1])
    #         new_swap_list.append(sorteddata[0][1])
    # elif third == 0:
    #     new_swap_list.append(sorteddata[2][1])
    #     if first == 1:
    #         new_swap_list.append(sorteddata[0][1])
    #         new_swap_list.append(sorteddata[1][1])
    #     elif first == 2:
    #         new_swap_list.append(sorteddata[1][1])
    #         new_swap_list.append(sorteddata[0][1])
    #     elif first == 3:
    #         new_swap_list.append(sorteddata[1][1])
    #         new_swap_list.append(sorteddata[0][1])
    # elif fourth == 0:
    #     new_swap_list.append(sorteddata[3][1])
    #     if first == 1:
    #         new_swap_list.append(sorteddata[0][1])
    #         if second == 2:
    #             new_swap_list.append(sorteddata[1][1])
    #             new_swap_list.append(sorteddata[2][1])
    #     elif first == 2:
    #         new_swap_list.append(sorteddata[0][1])
    #         if second == 1:
    #             new_swap_list.append(sorteddata[1][1])
    #             new_swap_list.append(sorteddata[2][1])

    # if minust[1] < 0:  #
    #     new_swap_list.append(sorteddata[1][1]) # leftmost = 2nd largest area
    #     new_swap_list.append(sorteddata[0][1])
    # else:
    #     new_swap_list.append(sorteddata[0][1])
    #     new_swap_list.append(sorteddata[1][1])
    # print("sorted data11[1]1", sorteddata[0][1])
    # print("new_swap_list:", new_swap_list)
    # find the nth largest contour [n-1][1], in this case 2
    # three_contours.append(sorteddata[0][1])
    # three_contours.append(sorteddata[1][1])
    # secondlargestcontour = sorteddata[1][1]
    print(">>new_swap_list", new_swap_list)
    # print(">>two_contours", two_contours)
    for c in new_swap_list:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        area = cv2.contourArea(c)

        if cv2.contourArea(c) < 100:
            continue

        box = approx
        box = np.squeeze(box)

        # order the points in the contour and draw outlines of the rotated rounding box
        box = order_points(box)

        print("box 1111", box)
        box = perspective.order_points(box)
        testing.append(box)
        # print("box 2222:", box)
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
    print("leftmost contour:", leftmost_contour)
    print("corners", corners)
    print("corners[1]", corners[1][0][0])
    # cv2.circle(threshold_image, (corners[1][0][0], corners[1][0][1]), 1, (100, 100, 100), thickness=10, lineType=8, shift=0)

    # cent = midpoint(center_points[1], center_points[2])
    # print("center of center:, ", cent)
    print("box testing", testing)
    testing2.append(testing[1])
    print("testing[1]", testing[1][0][0])
    testing2.append(testing[2])
    testing2.append(testing[3])
    testing2.append(testing[0])
    print("testing2", testing2)
    cv2.circle(threshold_image, (testing[1][0][0], testing[1][0][1]), 1, (100, 100, 100), thickness=10, lineType=8,
               shift=0)
    cv2.imshow("First corner?", threshold_image)
    for i in range(0, 4):
        print("w1", corners[i][2][0] - corners[i][3][0])
        print("w2", corners[i][1][0] - corners[i][0][0])
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
    cv2.circle(threshold_image, tuple(leftmost_contour[0][0]), 1, (100, 100, 100), thickness=10, lineType=8, shift=0)

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


def compute_reference(reference_marker_image):
    leftmost_contour, center_point, areas, distances, corners = grab_contour(reference_marker_image)
    # print("leftmost_[0][0]", leftmost_contour[0][0])
    # cv2.circle(reference_marker_image, tuple(leftmost_contour[0][0]), 1, (100, 100, 100), thickness=20, lineType=8, shift=0)

    # cv2.imshow("testing_leftmost", reference_marker_image)
    homography_points = []
    print("====== Reference Image ======")
    for i in range(0, 4):
        for j in range(0, 4):
            cv2.circle(reference_marker_image, tuple(corners[i][j]), 1, (100, 100, 100), thickness=2, lineType=8,
                       shift=0)
        cv2.circle(reference_marker_image, tuple(corners[0][0]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    # cv2.imshow("ref_corner_test", reference_marker_image)
    # corners
    # for i in range(0,2):
    #     for j in range(0,2):
    #         homography_points.append(corners[i][j])
    # print("homography points:", homography_points)
    # 불필요
    homography_points.append(corners[0][0])
    homography_points.append(corners[0][2])
    homography_points.append(corners[1][0])
    homography_points.append(corners[1][2])
    homography_points.append(corners[2][0])
    homography_points.append(corners[2][2])
    first_corners = corners[:len(corners) // 2]
    second_corners = corners[len(corners) // 2:]
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
    for i in range(0, 4):
        cv2.circle(reference_marker_image, (center_point[i]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    cv2.imshow("reference_marker_image", reference_marker_image)
    return leftmost_contour, center_point, corners, distances, homography_points


def compute_moved(moved_marker_image):
    leftmost_contour, center_point, areas, distances, corners = grab_contour(moved_marker_image)
    homography_points, corner_differences = [], []
    print("====== Moved Image ======")
    # print("ref_corners", corners)
    for i in range(0, 4):
        for j in range(0, 4):
            cv2.circle(moved_marker_image, tuple(corners[i][j]), 1, (100, 100, 100), thickness=2, lineType=8,
                       shift=0)
        # cv2.circle(moved_marker_image, tuple(ref_corners[0][0]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    # cv2.imshow("moved_corner_test", moved_marker_image)
    # corners
    # for i in range(0,2):
    #     for j in range(0,2):
    #         homography_points.append(corners[i][j])
    #         print("i", i)
    #         print("j", j)
    homography_points.append(corners[0][0])
    homography_points.append(corners[0][2])
    homography_points.append(corners[1][0])
    homography_points.append(corners[1][2])
    homography_points.append(corners[2][0])
    homography_points.append(corners[2][2])
    # print("corners", corners)

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
    print("w1", w1[1])
    corner_differences.append(np.array(first_corners) - np.array(second_corners))

    # print("moved corner differences", corner_differences)
    # distance of each marker
    # print("distance:", '{:.4}'.format(distances))
    print("Reference distances:", distances)

    # center points
    print("Moved center points:", center_point)

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
    print("Moved area:", areas)

    # distance between markers
    dist_markers = math.sqrt(
        (center_point[0][0] - center_point[1][0]) ** 2 + (center_point[0][1] - center_point[1][1]) ** 2)
    print("Distance between markers in moved markers top", dist_markers)
    dist_markers = math.sqrt(
        (center_point[1][0] - center_point[3][0]) ** 2 + (center_point[1][1] - center_point[3][1]) ** 2)
    print("Distance between markers in moved markers right", dist_markers)
    dist_markers = math.sqrt(
        (center_point[2][0] - center_point[3][0]) ** 2 + (center_point[2][1] - center_point[3][1]) ** 2)
    print("Distance between markers in moved markers bottom", dist_markers)
    dist_markers = math.sqrt(
        (center_point[0][0] - center_point[2][0]) ** 2 + (center_point[0][1] - center_point[2][1]) ** 2)
    print("Distance between markers in moved markers left", dist_markers)
    for i in range(0,4):
        cv2.circle(moved_marker_image, (center_point[i]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)
    cv2.imshow("Moved_marker_image", moved_marker_image)
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
    reference_marker_image = cv2.imread('images/1016_15m_transL_0.jpg')
    moved_marker_image = cv2.imread('images/1016_15m_transL_10.jpg')
    # overlay_image = cv2.imread('images/pig.png')

    reference_marker_image = apply_threshold(resize_image(reference_marker_image))
    moved_marker_image = apply_threshold(resize_image(moved_marker_image))

    # cv2.imshow("ref marker", reference_marker_image)
    # cv2.imshow("moved marker", moved_marker_image)
    ref_leftmost, ref_center, ref_corners, ref_distances, ref_homography_points = compute_reference(reference_marker_image)
    moved_leftmost, moved_center, moved_corners, moved_distances, moved_homography_points = compute_moved(moved_marker_image)
    angle_computation = []

    angle_difference, center_difference, corners_difference, distance_difference, homography_corners = [], [], [], [], []
    center_testing, angle_difference_reverse = [], []
    # for i in range(0, 6):
    #     cv2.circle(moved_marker_image, tuple(moved_homography_points[i]), 1, (100, 100, 100), thickness=5, lineType=8, shift=0)
    #
    # # cv2.imshow("ccc", moved_marker_image)
    #
    # for i in range(0, 6):
    #     cv2.circle(reference_marker_image, tuple(ref_homography_points[i]), 1, (100, 100, 100), thickness=3, lineType=8, shift=0)

    # cv2.imshow("ref_marker_img", reference_marker_image)
    # moved_homography_points = np.asarray(moved_homography_points, dtype=np.float32)
    # ref_homography_points = np.asarray(ref_homography_points, dtype=np.float32)
    #
    # un_warped = unwarp(moved_marker_image, moved_homography_points, ref_homography_points)

    print("====== Differences ======")

    for i in range(0, len(ref_center)):
        # center angle change
        angle_difference.append(angle_change(ref_center[i], moved_center[i]))
        angle_difference_reverse.append(angle_change(moved_center[i], ref_center[i]))
        # center point change
        center_difference.append(math.sqrt(
            (ref_center[i][0] - moved_center[i][0]) ** 2 + (ref_center[i][1] - moved_center[i][1]) ** 2))

        # corner points change
        corner_difference = np.stack(moved_corners) - np.stack(ref_corners)

        # markers' camera distance difference
        distance_difference.append(ref_distances[i] - moved_distances[i])

        print("ref center:", ref_center)
        # angle_computation.append(getAngle(ref_center[0], ref_center[1], ref_center[2]))
        # angle_computation.append(getAngle(ref_center[1], ref_center[2], ref_center[0]))
        # angle_computation.append(getAngle(ref_center[2], ref_center[0], ref_center[1]))
        # angle_computation.append(getAngle(moved_center[0], moved_center[1], moved_center[2]))
        # angle_computation.append(getAngle(moved_center[1], moved_center[2], moved_center[0]))
        # angle_computation.append(getAngle(moved_center[2], moved_center[0], moved_center[1]))
        print("ref 0 1 3", getAngle(ref_center[0], ref_center[1], ref_center[3]))
        print("ref 1 3 2", getAngle(ref_center[1], ref_center[3], ref_center[2]))
        print("ref 3 2 0", getAngle(ref_center[3], ref_center[2], ref_center[0]))
        print("ref 2 0 1", getAngle(ref_center[2], ref_center[0], ref_center[1]))
        print("moved 0 1 3", getAngle(moved_center[0], moved_center[1], moved_center[3]))
        print("moved 1 3 2", getAngle(moved_center[1], moved_center[3], moved_center[2]))
        print("moved 3 2 0", getAngle(moved_center[3], moved_center[2], moved_center[0]))
        print("moved 2 0 1", getAngle(moved_center[2], moved_center[0], moved_center[1]))
        # print("ref_Center[i][0]", ref_center[i][0])
        # print("ref center[i][1]", ref_center[i][1])
        print("moved center[i][0]", moved_center[i][0])
        print("moved center[i][1]", moved_center[i][1])
        center_testing.append(ref_center[i][0] - moved_center[i][0])
        center_testing.append(ref_center[i][1] - moved_center[i][1])
    # print("angle_Comput", angle_computation)
    # center_testing = np.stack(ref_center[i][0] - moved_center[i][0])
    # center_testing = np.stack(ref_center[i][1] - moved_center[i][1])
    print("Angle difference:", angle_difference)
    print("Angle difference reverse:", angle_difference_reverse)
    print("Center difference:", center_difference)
    print("Marker distance difference:", distance_difference)
    # print("Corner difference:", corner_difference)
    print("Center testing:", center_testing)

    # destination_points, h, w = get_destination_points(box)
    # overlay_image(moved_marker_image, overlay_image, moved_center[0][0], moved_center[0][1], angle, percentage)


if __name__ == '__main__':
    marker_calculation()
    cv2.waitKey()
    cv2.destroyAllWindows
