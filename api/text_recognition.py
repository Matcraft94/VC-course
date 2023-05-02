# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import pytesseract
from pytesseract import Output

import numpy as np

import matplotlib.pyplot as plt

# Funciones para procesar la imagen
def east_text_detection(image, min_confidence=0.5, width=320, height=320):
    # ... (Igual que en el ejemplo anterior)
    return boxes

def decode_predictions(scores, geometry, min_confidence):
    # ... (Igual que en el ejemplo anterior)
    return rects, confidences

def crnn_text_recognition(image):
    model_path = 'crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Cargar el modelo CRNN
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    # Preparar la imagen
    image = Image.fromarray(image).convert('L')
    image = transform(image)
    image = image.view(1, *image.size())
    image = Variable(image)

    # Realizar inferencia
    with torch.no_grad():
        output = model(image)
        _, preds = output.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

    # Decodificar el resultado
    result = []
    for p in preds:
        if p != 0 and (not result or result[-1] != p):
            result.append(alphabet[p - 1])

    return ''.join(result)

def draw_boxes_and_extract_text(image, boxes, model='tesseract'):
    for box in boxes:
        startX, startY, endX, endY = box
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        if model != 'east':
            roi = image[startY:endY, startX:endX]

            if model == 'tesseract':
                text = pytesseract.image_to_string(roi, lang="eng")
            elif model == 'crnn':
                text = crnn_text_recognition(roi)

            print(text)

    return image