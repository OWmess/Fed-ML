import cv2
import numpy as np
from skimage import measure
import torch
import onnxruntime
import time

enable_picamera=True
try :
    from picamera2 import Picamera2
except:
    enable_picamera=False

def vertical_projection(image):
    """计算并返回图像的垂直投影"""
    return np.sum(image, axis=0)

def find_peaks(projection, threshold=5, min_distance=5):
    """找到投影中的峰值，这些峰值表示可能的字符边界"""
    peaks = []
    for i in range(1, len(projection) - 1):
        if projection[i] > threshold and projection[i] > projection[i - 1] and projection[i] > projection[i + 1]:
            if not peaks or i - peaks[-1] > min_distance:
                peaks.append(i)
    return peaks

def segment_characters(image, vertical_proj):
    """使用垂直投影分割字符"""
    peaks = find_peaks(vertical_proj, threshold=np.max(vertical_proj) * 0.2, min_distance=10)
    characters = []
    for i in range(len(peaks) - 1):
        char_img = image[:, peaks[i]:peaks[i+1]]
        if char_img.shape[1] > 0:
            characters.append(char_img)
    return characters

if enable_picamera:
    picam2 = Picamera2()
    dispW = 1280
    dispH = 720
    picam2.preview_configuration.main.size = (dispW, dispH)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.controls.FrameRate = 30
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

roi_pts = []
selecting = False

def select_roi(event, x, y, flags, param):
    global selecting, roi_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(x, y)]
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            roi_pts[1:] = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts[1:] = [(x, y)]
        selecting = False

if __name__ == '__main__':
    ort_session = onnxruntime.InferenceSession("../models/mnist_model_0.onnx")
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', select_roi)
    roi_mode = False

    while True:
        frame=None
        if enable_picamera:
            frame = picam2.capture_array()
        else:

            cap = cv2.VideoCapture(0)
            cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            _,frame=cap.read()


        if frame.shape[0] != 720 or frame.shape[1] != 1280:
            frame = cv2.resize(frame, (1280, 720))
        if len(roi_pts) == 2 and not selecting:
            frame = frame[roi_pts[0][1]:roi_pts[1][1], roi_pts[0][0]:roi_pts[1][0]]
            roi_mode = True
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 5)
        thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh_img = cv2.bitwise_not(thresh_img)

        if roi_mode:
            v_proj = vertical_projection(thresh_img)
            characters = segment_characters(thresh_img, v_proj)

            for char in characters:
                x, y, w, h = cv2.boundingRect(char)
                digit = char[y:y + h, x:x + w]
                digit = cv2.resize(digit, (20, 20))
                mnist_digit = np.zeros((28, 28), dtype=np.float32)
                mnist_digit[4:24, 4:24] = digit
                mnist_digit = mnist_digit.reshape(1, 1, 28, 28)
                output_data = ort_session.run([output_name], {input_name: mnist_digit})[0]
                prediction = np.argmax(output_data)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(prediction), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('image', frame)

        key = cv2.waitKey(1)
        # 键盘输入q退出
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
