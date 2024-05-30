import cv2
import numpy as np
from skimage import measure
import torch
import onnxruntime
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

enable_picamera = True
SPLIT_THRESH = 10

try:
    from picamera2 import Picamera2
except:
    enable_picamera = False


def vertical_projection(image):
    """计算并返回图像的垂直投影"""
    projection = np.count_nonzero(image == 255, axis=0)
    return projection


def segment_characters(image, vertical_proj, threshold):
    """使用垂直投影分割字符"""
    lefts = []
    rights = []
    is_char = False
    for i in range(len(vertical_proj)):
        if vertical_proj[i] > threshold and not is_char:
            lefts.append(i)
            is_char = True
        elif vertical_proj[i] <= threshold and is_char:
            rights.append(i)
            is_char = False

    bounds = []  # 保存字符边界
    for left, right in zip(lefts, rights):
        char_img = image[:, left:right]
        if char_img.shape[1] > 0:
            bounds.append((left, right))  # 保存字符的左右边界

    return bounds


def pad_to_square(image):
    """将图像填充为正方形"""
    height, width = image.shape
    if height > width:
        pad = (height - width) // 2
        padded_img = np.pad(image, ((0, 0), (pad, pad)), 'constant', constant_values=0)
    elif width > height:
        pad = (width - height) // 2
        padded_img = np.pad(image, ((pad, pad), (0, 0)), 'constant', constant_values=0)
    else:
        padded_img = image
    return padded_img


def recognize_characters(bounds, image, ort_session, input_name, output_name):
    """识别字符并返回结果"""
    results = []
    # 先对bounds进行筛选，若字符长度小于最大字符长度的1/4则删除
    max_len = max([right - left for left, right in bounds])
    bounds = [(left, right) for left, right in bounds if right - left > max_len // 4]

    for left, right in bounds:
        char_img = image[:, left:right]
        # 寻找最大轮廓
        contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        char_img = char_img[y:y + h, x:x + w]
        char_img = pad_to_square(char_img)
        char_img = cv2.resize(char_img, (20, 20))

        mnist_digit = np.zeros((28, 28), dtype=np.float32)
        mnist_digit[4:24, 4:24] = char_img
        mnist_digit = mnist_digit.reshape(1, 1, 28, 28)

        ort_inputs = {input_name: mnist_digit}
        ort_outs = ort_session.run([output_name], ort_inputs)
        pred = np.argmax(ort_outs[0])
        results.append((left, right, pred))
        cv2.imshow('digit', mnist_digit[0, 0] * 255)
        cv2.waitKey()
    return results


def vertical_projection_show(vertical_proj):
    """创建带滑块的垂直投影图"""
    global SPLIT_THRESH

    def update(val):
        global SPLIT_THRESH
        SPLIT_THRESH = slider.val
        plt.draw()

    plt.switch_backend('tkagg')
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.bar(range(len(vertical_proj)), vertical_proj)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Thresh', 0, 255, valinit=SPLIT_THRESH, valstep=1)

    slider.on_changed(update)
    plt.show()


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
    ort_session = onnxruntime.InferenceSession("../models/mnist_model.onnx")
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', select_roi)
    roi_mode = False
    cap = None
    if not enable_picamera:
        cap = cv2.VideoCapture(-1)
        cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    while True:
        frame = None

        if enable_picamera:
            frame = picam2.capture_array()
        else:
            _ , frame = cap.read()
            pass

        # frame=cv2.imread("../tools/captured_image.jpg")

        if frame.shape[0] != 720 or frame.shape[1] != 1280:
            frame = cv2.resize(frame, (1280, 720))
        if len(roi_pts) == 2 and not selecting:
            frame = frame[roi_pts[0][1]:roi_pts[1][1], roi_pts[0][0]:roi_pts[1][0]]
            roi_mode = True
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 5)
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,
                                           3)  # 自适应二值化
        thresh_img = cv2.bitwise_not(thresh_img)
        # 膨胀操作
        kernel = np.ones((11, 11), np.uint8)
        thresh_img = cv2.dilate(thresh_img, kernel, iterations=3)
        if roi_mode:
            v_proj = vertical_projection(thresh_img)
            vertical_projection_show(v_proj)
            bounds = segment_characters(thresh_img, v_proj, SPLIT_THRESH)

            # 识别字符
            results = recognize_characters(bounds, thresh_img, ort_session, input_name, output_name)

            # 在frame原图像中框选出字符并显示识别结果
            for left, right, pred in results:
                cv2.rectangle(frame, (left, 0), (right, frame.shape[0]), (0, 255, 0), 2)
                cv2.putText(frame, str(pred), (left, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # 将frame和thresh_img拼接
        combined_img = np.hstack((frame, cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('image', combined_img)
        key = cv2.waitKey(1)
        # 键盘输入q退出
        if key == ord('q'):
            break
        elif roi_mode:
            cv2.waitKey()
            exit(0)

    cv2.destroyAllWindows()
