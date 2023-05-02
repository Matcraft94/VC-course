# Creado por: Lucy
# Fecha de creación: 30/04/2023

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2

import numpy as np

from api.segmentation import *

def main():
    # Inicializar la captura de video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error al abrir la cámara.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        libros_segmentados, mascaras_binarias = segmentar_libros(frame)

        # Aplicar máscaras de color a los libros segmentados
        for mascara_binaria in mascaras_binarias:
            color = tuple(random.randint(0, 255) for _ in range(3))
            frame = aplicar_mascara_color(frame, mascara_binaria, color)

        # Mostrar la cantidad total de libros en pantalla
        cv2.putText(frame, f"Total de libros: {len(libros_segmentados)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostrar la imagen con los libros segmentados y enumerados
        cv2.imshow("Libros en tiempo real", frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de video y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Leer el primer fotograma y segmentar los libros
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el primer fotograma.")
        return

    libros_segmentados, mascaras_binarias = segmentar_libros(frame)

    # Inicializar el rastreador multilobjeto
    tipo_rastreador = "KCF"  # Puedes cambiar esto a "KCF", "MIL", "TLD", "BOOSTING", "MOSSE" o "CSRT"
    rastreador_multilobjeto = inicializar_rastreador(tipo_rastreador, frame, libros_segmentados)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Actualizar el rastreador multilobjeto y obtener los nuevos rectángulos de los libros
        _, rectangulos_actualizados = rastreador_multilobjeto.update(frame)

        # Dibujar rectángulos actualizados y enumerar los libros
        for i, rect in enumerate(rectangulos_actualizados):
            x, y, w, h = [int(v) for v in rect]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Libro {i + 1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Segmentar los libros
        libros_segmentados, mascaras_binarias = segmentar_libros(frame)

        # Mostrar los libros segmentados con máscaras de colores
        for i, mascara_binaria in enumerate(mascaras_binarias):
            color = np.random.randint(0, 256, 3).tolist()
            frame = aplicar_mascara_color(frame, mascara_binaria, color)

        # Reconocer los títulos de los libros
        titulos = reconocer_titulos(libros_segmentados, frame)

        # Mostrar los títulos de los libros
        for i, (x, y, w, h) in enumerate(libros_segmentados):
            cv2.putText(frame, titulos[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Libros en tiempo real", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    cap = cv2.VideoCapture(0)

    # Inicializa el modelo de segmentación
    modelo_segmentacion = inicializar_modelo_segmentacion()

    # Configuración del rastreador
    tipo_rastreador = "MIL"

    rastreador_multilobjeto = None
    titulos = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rastreador_multilobjeto is None:
            libros_segmentados = segmentar_libros(modelo_segmentacion, frame)

            # Aplicar máscara de color a los libros segmentados
            color = (0, 255, 0)
            for i, libro in enumerate(libros_segmentados):
                frame = aplicar_mascara_color(frame, libro, color)

            if len(libros_segmentados) > 0:
                rastreador_multilobjeto = inicializar_rastreador(tipo_rastreador, frame, libros_segmentados)
                titulos = reconocer_titulos(libros_segmentados, frame)
        else:
            libros_rastreados = []
            for rastreador in rastreador_multilobjeto:
                success, bbox = rastreador.update(frame)
                if success:
                    libros_rastreados.append(bbox)

            for i, bbox in enumerate(libros_rastreados):
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                cv2.putText(frame, titulos[i], (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Libros en tiempo real", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
