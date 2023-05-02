# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import numpy as np

import matplotlib.pyplot as plt

def image_prepross(img: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.resize(img_gray, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

    area = img_gray.shape[0] * img_gray.shape[1]

    if area < 100000:
        i, j = 55, 21
    elif area < 300000:
        i, j = 57, 23
    else:
        i, j = 59, 25

    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, i, j)

    return img_gray