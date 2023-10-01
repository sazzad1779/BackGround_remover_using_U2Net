import sys
import os
import argparse
import cv2
import numpy as np
from PIL import Image

def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        print(f'{filename} not found')
        sys.exit()

def get_capture(video):

    try:
        video_id = int(video)

        # webcamera-mode
        capture = cv2.VideoCapture(video_id)
        if not capture.isOpened():
            print(f"[ERROR] webcamera (ID - {video_id}) not found")
            sys.exit(0)

    except ValueError:
        # if file path is given, open video file
        if check_file_existance(video):
            capture = cv2.VideoCapture(video)

    return capture

def format_input_tensor(tensor, input_details, idx):
    details = input_details[idx]
    dtype = details['dtype']
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details['quantization_parameters']
        input_tensor = tensor / quant_params['scales'] + quant_params['zero_points']
        if dtype == np.int8:
            input_tensor = input_tensor.clip(-128, 127)
        else:
            input_tensor = input_tensor.clip(0, 255)
        return input_tensor.astype(dtype)
    else:
        return tensor.astype('float32')

def imread(filename, flags=cv2.IMREAD_COLOR):
    if not os.path.isfile(filename):
        print(f"File does not exist: {filename}")
        sys.exit()
    data = np.fromfile(filename, np.int8)
    img = cv2.imdecode(data, flags)
    return img

def transform(image, scaled_size):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = scaled_size[1]*h/w, scaled_size[0]
    else:
        new_h, new_w = scaled_size[1], scaled_size[0]*w/h
    new_h, new_w = int(new_h), int(new_w)
    
    image = cv2.resize(image, (scaled_size[0], scaled_size[1]))

    # ToTensorLab part in original repo
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image/np.max(image)

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225
    return tmpImg[np.newaxis, :, :, :]


def load_image(image_path, scaled_size, rgb_mode):
    image = imread(image_path)
    if rgb_mode and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[0], image.shape[1]
    if 2 == len(image.shape):
        image = image[:, :, np.newaxis]
    return transform(image, scaled_size), h, w


def norm(pred):
    ma = np.max(pred)
    mi = np.min(pred)
    return (pred - mi) / (ma - mi)

def process_result(pred, srcimg_shape):
    pred = cv2.resize(pred[0], (srcimg_shape[1], srcimg_shape[0]))
    pred = norm(pred)
    return pred
