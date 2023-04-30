# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import pytesseract
from pytesseract import Output

import dlib

import numpy as np

import matplotlib.pyplot as plt

from api.preprocess import *
from api.document import *
from api.text import *

# Cargar el modelo de detección de caras de OpenCV y el predictor de puntos clave faciales de dlib
face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "res10_300x300_ssd_iter_140000 (1).caffemodel")
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Preparar la entrada para el modelo de detección de caras
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Realizar la detección de caras
    face_net.setInput(blob)
    detections = face_net.forward()

    # Analizar las detecciones y dibujar las caras y los puntos clave faciales
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            rect = dlib.rectangle(x1, y1, x2, y2)

            # Obtener las coordenadas (x, y) de los puntos clave faciales
            landmarks = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rect)
            landmarks_coords = np.zeros((68, 2), dtype=int)

            for i in range(0, 68):
                landmarks_coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

            # Dibujar los puntos clave faciales y un contorno alrededor de la cara
            for point in landmarks_coords:
                cv2.circle(frame, tuple(point), 1, (0, 0, 255), -1)

            # Extraer las coordenadas del contorno de la cara (puntos 0 a 16)
            jaw = landmarks_coords[:17]
            cv2.polylines(frame, [jaw], False, (0, 255, 0), 1)

    # Mostrar el frame actual
    cv2.imshow("Video en vivo", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
