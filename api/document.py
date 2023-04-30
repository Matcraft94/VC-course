# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import pytesseract
from pytesseract import Output

import numpy as np

import matplotlib.pyplot as plt

def reconocer_tipo_documento(texto):
    if 'CEDULA' in texto:
        return 'Tipo de documento: CEDULA'
    else:
        return 'Tipo de documento: Desconocido'


def detectar_documento(imagen):
    # Convertir a escala de grises y aplicar desenfoque gaussiano
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # Aplicar el algoritmo de Canny para detectar bordes
    bordes = cv2.Canny(gris, 75, 200)

    # Encontrar contornos en la imagen de bordes
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]

    # Buscar un contorno que tenga 4 vértices y una gran área
    for c in contornos:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return True, approx

    return False, None