# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import pytesseract
from pytesseract import Output

import numpy as np

import matplotlib.pyplot as plt

from api.preprocess import *
from api.document import *
from api.text import *

# Configurar pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Cambia esta ruta a la ubicación de tesseract en tu sistema


# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Verificar si hay un documento en pantalla
    hay_documento, vertices = detectar_documento(frame)

    if hay_documento:
        # Dibujar un rectángulo alrededor del documento
        cv2.drawContours(frame, [vertices], -1, (0, 255, 0), 2)

        # Realizar OCR en el frame
        datos_ocr = pytesseract.image_to_data(frame, lang="spa", config="--psm 6", output_type=Output.DICT)

        # Encuadrar y mostrar el texto reconocido
        encuadrar_y_mostrar_texto(frame, datos_ocr)

    # Mostrar el frame actual
    cv2.imshow("Video en vivo", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
