#神经网络识别手写字符串
import cv2
import numpy as np
from skimage import measure
import torch
import onnxruntime
import LeNet5
import time
import sys
import os
from picamera2 import Picamera2

# 使用OpenCV捕获视频
picam2 = Picamera2()
# init camera
dispW = 1280
dispH = 720
picam2.preview_configuration.main.size = (dispW, dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()





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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




# 初始化列表用于保存ROI的坐标
roi_pts = []
# 创建一个标记变量来指示是否正在选择ROI
selecting = False

# 定义鼠标事件的回调函数
def select_roi(event, x, y, flags, param):
    global selecting, roi_pts

    if event == cv2.EVENT_LBUTTONDOWN:
        # 当按下鼠标左键时，开始选择ROI并保存初始点
        roi_pts = [(x, y)]
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE:
        # 当鼠标移动时，更新ROI的结束点
        if selecting:
            roi_pts[1:] = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        # 当释放鼠标左键时，结束选择ROI
        roi_pts[1:] = [(x, y)]
        selecting = False






if __name__ == '__main__':
    image = cv2.imread('../tools/image2.jpg')
    ort_session = onnxruntime.InferenceSession("../models/mnist_model_0.onnx")
    input_name=ort_session.get_inputs()[0].name
    output_name=ort_session.get_outputs()[0].name
    #从摄像头获取图片
    exit_cnt=0
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', select_roi)
    roi_mode=False
    frame = None
    thresh_img=None

    while True:
        frame=picam2.capture_array()
        # frame=cv2.imread('../tools/image2.jpg')
        #如果图像分辨率不符合1280x720，resize
        if frame.shape[0]!=720 or frame.shape[1]!=1280:
            frame=cv2.resize(frame,(1280,720))
        if len(roi_pts) == 2 and not selecting:
            frame=frame[roi_pts[0][1]:roi_pts[1][1],roi_pts[0][0]:roi_pts[1][0]]
            roi_mode=True
        key=cv2.waitKey(1)

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_img=cv2.equalizeHist(gray_img)
        warped_img = frame
        #中值滤波
        gray_img=cv2.medianBlur(gray_img, 5)
        thresh_img=cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)#自适应二值化
        thresh_img=cv2.bitwise_not(thresh_img)#反转像素

        if roi_mode:
            #连通域去除噪点
            labels = measure.label(thresh_img, connectivity=2, background=0)
            mask = np.zeros(thresh_img.shape, dtype="uint8")
            for label in np.unique(labels):
                if label == 0:
                    continue
                labelMask = np.zeros(thresh_img.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)
                if numPixels > 10:
                    mask = cv2.add(mask, labelMask)
            thresh_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
            kernel = np.ones((5, 5), np.uint8)
            thresh_img=cv2.dilate(thresh_img,kernel)

        # cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
        # cv2.imshow('Threshold', thresh_img)
        if roi_mode:
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
                # 调整图像大小为20x20像素
                digit = cv2.resize(digit, (20, 20))

                # 创建一个空白的28x28像素图像
                mnist_digit = np.zeros((28, 28), dtype=np.float32)

                # 将20x20像素的图像放在中心
                mnist_digit[4:24, 4:24] = digit
                res=mnist_digit
                mnist_digit=mnist_digit.reshape(1,1,28,28)
                output_data = ort_session.run([output_name], {input_name: mnist_digit})[0]
                probabilities = np.exp(output_data)
                prediction = np.argmax(probabilities)
                print(prediction)
                # 在图像上画出边界框并添加预测的结果
                cv2.rectangle(warped_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(warped_img, str(prediction.item()), (int(x+w/2), int(y +h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # print(prediction)
            # print(output_data)
            # cv2.namedWindow('Digit', cv2.WINDOW_NORMAL)
            # cv2.imshow('Digit', res)
            # cv2.waitKey(0)

        image = np.hstack((warped_img, cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('image', image)

        if roi_mode:
            break
        #显示
        # cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
        # cv2.imshow('Contours', warped_img)

    time.sleep(1)
    cv2.waitKey()



    cv2.destroyAllWindows()
