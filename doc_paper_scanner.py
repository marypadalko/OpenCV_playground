import cv2
import numpy as np
# from experiments import stackImages


def preprocessing(img):
    """
    Detects edges and outputs only thick ones
    :param img: input image or video frame from webcam
    :return: img with only thick edges detected
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # edges
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    # make edges 'better', so that we ignore noise
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDil, kernel, iterations=1)
    return imgErode


def getContours(img, imgContour):
    """
    Get's the largest object's contour or just the largest closed contour
    :param img: input image
    :return: biggest rectangular contour coords
    """
    maxArea = 0
    biggest = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            perimeter = cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)
            objCorners = len(approx)  # should be 4
            if objCorners == 4 and area > maxArea:
                biggest = approx
                maxArea = area
    # now draw the contour of the largest rectangular object
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 3)
    return biggest


def reorder(points):
    """
    Sorts the array of points coordinates by their sum
    :param points: list of points's coordinates
    :return: reordered list of coords of points
    """
    points = points.reshape((4, 2))
    result = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    result[0] = points[np.argmin(add)]
    result[3] = points[np.argmin(add)]
    delta = np.diff(result, axis=1)
    result[1] = points[np.argmin(delta)]
    result[2] = points[np.argmax(delta)]
    return result


def getwarp(img, biggest):
    """
    Warps the largest rectangular object
    :param img: input image
    :param biggest: contour of the largest rectangular object
    :return: warps that object
    """
    # coordinates of the initial contour in correct order
    pts1 = np.float32(reorder(biggest))
    # target, we want to stretch that contour to the whole frame
    width, height = frameWidth, frameHeight
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # transformation to get that target
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # result
    imgOut = cv2.warpPerspective(img, matrix, (width, height))

    # let's crop a bit redundant noise
    imgCropped = imgOut[20:-20, 20:-20]
    imgResized = cv2.resize(imgCropped, (frameWidth, frameHeight))
    return imgResized


frameHeight = 480
frameWidth = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    imgContour = img.copy()
    imgThres = preprocessing(img)
    biggest = getContours(imgThres, imgContour)
    imgWarp = getwarp(img, biggest)
    cv2.imshow('Video frame', imgWarp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break