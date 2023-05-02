# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50

import pytesseract
from pytesseract import Output

import numpy as np

import matplotlib.pyplot as plt

# Procesar la imagen y realizar la segmentación
def process_image(image, model):
    input_tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

# Cargar el modelo de segmentación pre-entrenado
def load_model():
    model = deeplabv3_resnet50(pretrained=True, progress=True, num_classes=21, aux_loss=None)
    model.eval()
    return model