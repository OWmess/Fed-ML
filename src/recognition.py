#神经网络识别手写字符串
import cv2
import numpy as np

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
    image = cv2.imread('../tools/image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    warped = four_point_transform(image, screenCnt.reshape(4, 2))
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI", warped)
    cv2.waitKey(0)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
    cv2.imshow('Threshold', thresh)

    #投影法分割字符

    # 垂直投影
    (h, w) = thresh.shape
    sum_cols = []
    for j in range(w):
        col = thresh[:, j]
        sum_cols.append(np.sum(col)/255)

    # 寻找投影直方图中的最小值点进行分割
    th = max(sum_cols) // 2
    start_i = None
    intervals = []
    for i, sum_col in enumerate(sum_cols):
        if sum_col > th and start_i is None:
            start_i = i
        if sum_col <= th and start_i is not None:
            end_i = i
            intervals.append((start_i, end_i))
            start_i = None

    # 显示分割结果
    for interval in intervals:
        start_i, end_i = interval
        char_img = thresh[:, start_i:end_i]
        cv2.namedWindow('Char', cv2.WINDOW_NORMAL)
        cv2.imshow('Char', char_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
