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
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
        frame_g = image_prepross(frame)
        texto_ocr = pytesseract.image_to_string(frame_g, lang="spa", config="--psm 6")
        tipo_documento = reconocer_tipo_documento(texto_ocr)
        print(tipo_documento)

    # Mostrar el frame actual
    cv2.imshow("Video en vivo", frame)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.pause(0.01)  # Agrega una pausa para actualizar la imagen
    # plt.clf()  # Limpia el gráfico antes de la siguiente iteración

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
    # if plt.waitforbuttonpress(0.001):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
