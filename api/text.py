# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import pytesseract
from pytesseract import Output

import numpy as np

import matplotlib.pyplot as plt

def encuadrar_y_mostrar_texto(imagen, datos):
    for i in range(len(datos["level"])):
        if datos["level"][i] == 5:  # Nivel de palabra
            x, y, w, h = datos["left"][i], datos["top"][i], datos["width"][i], datos["height"][i]
            palabra = datos["text"][i]

            # Dibujar un rectángulo alrededor de la palabra
            cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Mostrar el texto reconocido encima del rectángulo
            cv2.putText(imagen, palabra, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

