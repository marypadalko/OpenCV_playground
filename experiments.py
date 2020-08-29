# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np


def show_lena(title='lena'):
    # Use a breakpoint in the code line below to debug your script.
    img = cv2.imread("lena.png")
    print(type(img))
    cv2.imshow(title, img)
    cv2.waitKey(1000)


def cap_video():
    cap = cv2.VideoCapture('video_mp4.mp4')
    success = True
    while success:
        success, img = cap.read()
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def cap_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set width
    cap.set(4, 480)  # set height
    cap.set(10, 300)  # set brightness
    while True:
        success, img = cap.read()
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def basic_transforms():
    img = cv2.imread('lena.png')

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray scale', imgGray)
    cv2.waitKey(500)

    imgBlur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow('blurred', imgBlur)
    cv2.waitKey(500)

    imgCanny = cv2.Canny(img, threshold1=100, threshold2=100)
    cv2.imshow('canny edge', imgCanny)
    cv2.waitKey(500)

    imgDilation = cv2.dilate(imgCanny, np.ones((5, 5), dtype=np.uint8), iterations=1)
    cv2.imshow('dilation', imgDilation)
    cv2.waitKey(500)

    imgErosion = cv2.erode(imgDilation, np.ones((5, 5), dtype=np.uint8), 1)
    cv2.imshow('erosion', imgErosion)
    cv2.waitKey(500)


def sizes():
    img = cv2.imread('lena.png')
    # print(img.shape)

    imgResize = cv2.resize(img, (200, 200))
    print(imgResize.shape)
    cv2.imshow('img', imgResize)
    cv2.waitKey(500)

    imgCropped = img[100:300, 100:300, :] # height, width, ch
    cv2.imshow('imgCropped', imgCropped)
    cv2.waitKey(500)


def shapes_and_text():
    img = np.zeros((512, 512, 3), np.uint8)
    # img[:] = 255, 0, 0   # all blue since BGR
    cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
    cv2.rectangle(img, (0, 0), (250, 350), (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (400, 50), 30, (255, 0, 0), 5)
    cv2.putText(img, 'OpenCV', (300, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (1, 150, 1), 1)
    cv2.imshow('geom', img)
    cv2.waitKey(5000)


def warp_perspective():
    img = cv2.imread('cards.jpg')
    width, height = 250, 350  # this will be dims of the card
    pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
    # pts1 = coordinates of a specific card in the pic
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # pts2 = coordinates of f(pts1)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOut = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imshow('imgOut', imgOut)
    cv2.waitKey(500)


def join_images():
    img = cv2.imread('lena.png')
    hor_stack = np.hstack((img, img))  # or vstack
    cv2.imshow('hor_stack', hor_stack)
    cv2.waitKey(500)




def color_stuff():

    """
    takes orange pixels
    :return: just shows pic with orange pixels and track bar :)
    """

    def empty(x):
        return x

    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

    while True:
        img = cv2.imread('lena.png')
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])  # lower limit
        upper = np.array([h_max, s_max, v_max])  # upper limit
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("Original",img)
        # cv2.waitKey(500)
        # cv2.imshow("HSV",imgHSV)
        # cv2.waitKey(500)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(500)
        cv2.imshow("Result", imgResult)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print('area', area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, closed=True)
            print('perimeter', perimeter)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)
            objCorners = len(approx)
            print('corners 0.02', objCorners)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if objCorners == 3:
                objType = 'triangle'
            elif objCorners == 4:
                aspRatio = w/float(h)
                if 0.95 <= aspRatio <= 1.05:
                    objType = 'square'
                else:
                    objType = 'rectangle'
            elif objCorners > 4:
                objType = 'circle'

            cv2.putText(imgContour, objType, (x + w//2 - 10,
                        y + h//2 -10), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)


def contours_shape_detection():
    img = cv2.imread('shapes.png')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    imgContour = img.copy()
    getContours(imgCanny, imgContour)

    imgStacked = stackImages(0.5, [img, imgBlur, imgCanny, imgContour])
    cv2.imshow('imgStacked', imgStacked)
    cv2.waitKey(0)


def face_detection():
    img = cv2.imread('lena.png')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(125,125,0),2)
        cv2.imshow('img', img)
        cv2.waitKey(0)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # show_lena()
    # cap_video()
    # cap_webcam()
    # basic_transforms()
    # sizes()
    # shapes_and_text()
    # warp_perspective()
    # join_images()
    # color_stuff()
    # contours_shape_detection()
    face_detection()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
