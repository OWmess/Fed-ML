#神经网络识别手写字符串
import cv2
import numpy as np
from skimage import measure
import torch
import onnxruntime
import train_mnist
import time
import sys
import os
# from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
# from qtpy.QtGui import QPixmap,QImage
# from qtpy import PYQT6_API


#
# class AppWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Image Display UI")
#         self.setGeometry(300, 300, 800, 450)
#
#         # Create a QWidget and set it as central widget
#         widget = QWidget(self)
#         self.setCentralWidget(widget)
#
#         # Create a vertical layout and add widgets
#         layout = QVBoxLayout()
#         widget.setLayout(layout)
#
#         # Create Labels for original and binary images
#         self.label_original = QLabel()
#         self.label_binary = QLabel()
#
#         # Add labels to layout
#         layout.addWidget(self.label_original)
#         layout.addWidget(self.label_binary)
#
#     def display_images(self, original_image_path, binary_image_path):
#         # Load images
#         original_image = cv2.imread(original_image_path)
#         binary_image = cv2.imread(binary_image_path)
#
#         # Convert images to format suitable for QPixmap
#         original_image = self.convert_cv_qt(original_image)
#         binary_image = self.convert_cv_qt(binary_image)
#
#         # Show images on labels
#         self.label_original.setPixmap(QPixmap.fromImage(original_image))
#         self.label_binary.setPixmap(QPixmap.fromImage(binary_image))
#
#     def convert_cv_qt(self, cv_img):
#         rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         p = convert_to_qt_format.scaled(400, 400, PYQT6_API.QtCore.Qt.KeepAspectRatio)
#         return QPixmap.fromImage(p)

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


if __name__ == '__main__':
    image = cv2.imread('../tools/image2.jpg')
    ort_session = onnxruntime.InferenceSession("../models/mnist_model_0.onnx")
    input_name=ort_session.get_inputs()[0].name
    output_name=ort_session.get_outputs()[0].name
    #从摄像头获取图片
    cap=cv2.VideoCapture(0)
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    exit_cnt=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame . Retry ...")
            exit_cnt+=1
            if exit_cnt>20:
                os._exit(0)
            time.sleep(1)
            continue

        key=cv2.waitKey(1)
        if key =='q':
            break


        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        warped_img = frame

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
            # 调整图像大小为20x20像素
            digit = cv2.resize(digit, (20, 20))

            # 创建一个空白的28x28像素图像
            mnist_digit = np.zeros((28, 28), dtype=np.float32)

            # 将20x20像素的图像放在中心
            mnist_digit[4:24, 4:24] = digit
            res=mnist_digit
            mnist_digit=mnist_digit.reshape(1,1,28,28)
            # if key== 's':
            output_data = ort_session.run([output_name], {input_name: mnist_digit})[0]
            probabilities = np.exp(output_data)
            prediction = np.argmax(probabilities)
            print(prediction)
            # 在图像上画出边界框并添加预测的结果
            cv2.rectangle(warped_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(warped_img, str(prediction.item()), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                # cv2.waitKey()
            # print(prediction)
            # print(output_data)
            # cv2.namedWindow('Digit', cv2.WINDOW_NORMAL)
            # cv2.imshow('Digit', res)
            # cv2.waitKey(0)



        #显示
        cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
        cv2.imshow('Contours', warped_img)

        # app = QApplication(sys.argv)
        #
        # window = AppWindow()
        # window.display_images('path_to_original_image', 'path_to_binary_image')
        # window.show()
        #
        # sys.exit(app.exec_())

    cv2.destroyAllWindows()
