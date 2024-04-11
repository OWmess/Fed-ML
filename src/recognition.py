#神经网络识别手写字符串
import cv2
import numpy as np
from skimage import measure
import torch
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

if __name__ == '__main__':
    image = cv2.imread('../tools/image2.jpg')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edged = cv2.Canny(gray_img, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    warped_img = image
    if not screenCnt is None:
        warped_img = four_point_transform(image, screenCnt.reshape(4, 2))
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.imshow("ROI", warped_img)
        cv2.waitKey(0)

    gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    #中值滤波
    gray_img=cv2.medianBlur(gray_img, 3)
    thresh_img=cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)#自适应二值化
    thresh_img=cv2.bitwise_not(thresh_img)#反转像素

    #连通域去除噪点
    labels = measure.label(thresh_img, connectivity=2, background=0)
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > 50:
            mask = cv2.add(mask, labelMask)
    thresh_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)


    cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
    cv2.imshow('Threshold', thresh_img)

    # 寻找图像中的所有轮廓
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # 对每一个找到的轮廓，计算其边界框并将其添加到边界框列表中
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]

    # 对边界框进行排序，排序基准为边界框的x坐标
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

    # 遍历每一个边界框，分割出其中的字符
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        digit = thresh_img[y:y + h, x:x + w]
        #resize为28*28的图像
        digit = cv2.resize(digit, (28, 28))
        # 调整图像大小为20x20像素
        digit = cv2.resize(digit, (20, 20))

        # 创建一个空白的28x28像素图像
        mnist_digit = np.zeros((28, 28), dtype="uint8")

        # 将20x20像素的图像放在中心
        mnist_digit[4:24, 4:24] = digit
        cv2.namedWindow('Digit', cv2.WINDOW_NORMAL)
        cv2.imshow('Digit', mnist_digit)
        cv2.waitKey(0)


    #显示
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.imshow('Contours', warped_img)


    cv2.waitKey()



    cv2.destroyAllWindows()
