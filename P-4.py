# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2

import numpy as np

from api.torch_process import *


# Mostrar la imagen segmentada en tiempo real
def display_segmentation():
    model = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        segmented = process_image(frame, model)

        # Crear una máscara de color para cada segmento y combinarla
        seg_color = cv2.applyColorMap(segmented * 10, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(frame, 0.5, seg_color, 0.5, 0)

        cv2.imshow('Segmented Video Feed', combined)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_segmentation()
