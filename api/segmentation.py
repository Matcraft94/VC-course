# Creado por: Lucy
# Fecha de creación: 30/04/2023

import random

import cv2

import easyocr

import numpy as np

import torch
from torchvision import models
from torchvision import transforms
import torchvision

import time


# def inicializar_modelo_segmentacion():
#     # Cargar el modelo de segmentación DeepLabV3 con ResNet50
#     modelo_segmentacion = models.segmentation.deeplabv3_resnet50(pretrained=True)
#     modelo_segmentacion.eval()

#     return modelo_segmentacion

# # def crear_rastreador_individual(tipo_rastreador):
# #     if tipo_rastreador == 'KCF':
# #         rastreador = cv2.TrackerKCF_create()
# #     elif tipo_rastreador == 'TLD':
# #         rastreador = cv2.TrackerTLD_create()
# #     elif tipo_rastreador == 'MIL':
# #         rastreador = cv2.TrackerMIL_create()
# #     elif tipo_rastreador == 'CSRT':
# #         rastreador = cv2.TrackerCSRT_create()
# #     elif tipo_rastreador == 'MOSSE':
# #         rastreador = cv2.TrackerMOSSE_create()
# #     else:
# #         raise ValueError("Tipo de rastreador no soportado. Por favor, elige entre 'KCF', 'TLD', 'MIL', 'CSRT', 'MOSSE'.")

# #     return rastreador

def inicializar_modelo_segmentacion():
    modelo = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained_backbone=True,
        progress=True,
        num_classes=21,
        aux_loss=None,
        pretrained=True#,
        # # usar weights en lugar de pretrained
        # weights="torchvision://deeplabv3_resnet50_coco"
    )
    # establecer el modelo en modo de evaluación
    modelo.eval()
    # mover el modelo a la GPU si está disponible
    if torch.cuda.is_available():
        modelo.cuda()
    return modelo

def crear_rastreador_individual(tipo_rastreador):
    opencv_version = cv2.__version__.split('.')[0]
    if opencv_version == '3':
        rastreadores = {
            "BOOSTING": cv2.TrackerBoosting_create,
            "MIL": cv2.TrackerMIL_create,
            "KCF": cv2.TrackerKCF_create,
            "TLD": cv2.TrackerTLD_create,
            "MEDIANFLOW": cv2.TrackerMedianFlow_create,
            "GOTURN": cv2.TrackerGOTURN_create,
            "MOSSE": cv2.TrackerMOSSE_create,
            "CSRT": cv2.TrackerCSRT_create
        }
    elif opencv_version == '4':
        rastreadores = {
            "BOOSTING": cv2.legacy_TrackerBoosting.create,
            "MIL": cv2.legacy_TrackerMIL.create,
            "KCF": cv2.legacy_TrackerKCF.create,
            "TLD": cv2.legacy_TrackerTLD.create,
            "MEDIANFLOW": cv2.legacy_TrackerMedianFlow.create,
            "GOTURN": cv2.legacy_TrackerGOTURN.create,
            "MOSSE": cv2.legacy_TrackerMOSSE.create,
            "CSRT": cv2.legacy_TrackerCSRT.create
        }
    else:
        raise ValueError("Versión de OpenCV no soportada")

    if tipo_rastreador not in rastreadores:
        raise ValueError(f"Tipo de rastreador no válido: {tipo_rastreador}")

    return rastreadores[tipo_rastreador]()



# def inicializar_rastreador(tipo_rastreador, frame, objetos):
#     rastreador_multilobjeto = cv2.MultiTracker()

#     for objeto in objetos:
#         x, y, w, h = objeto
#         rectangulo_objeto = (x, y, w, h)
#         rastreador_individual = crear_rastreador_individual(tipo_rastreador)
#         rastreador_multilobjeto.add(rastreador_individual, frame, rectangulo_objeto)

#     return rastreador_multilobjeto


def inicializar_rastreador(tipo_rastreador, frame, objetos_segmentados):
    rastreadores = []

    for objeto in objetos_segmentados:
        rastreador = crear_rastreador_individual(tipo_rastreador)
        rastreador.init(frame, objeto)
        rastreadores.append(rastreador)

    return rastreadores


def aplicar_mascara_color(frame, mascara_binaria, color):
    # Redimensionar la máscara binaria a las dimensiones del fotograma original
    altura, ancho = frame.shape[:2]
    mascara_binaria = cv2.resize(mascara_binaria, (ancho, altura))

    # Aplicar la máscara de color al fotograma
    mascara_color = np.zeros_like(frame)
    mascara_color[mascara_binaria == 255] = color
    frame_con_mascara = cv2.addWeighted(frame, 0.7, mascara_color, 0.3, 0)
    return frame_con_mascara


# # def segmentar_libros(frame):
# #     # Cargar el modelo de segmentación (aquí se utiliza un modelo pre-entrenado de torchvision)
# #     modelo_segmentacion = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
# #     modelo_segmentacion.eval()

# #     # Preprocesar la imagen
# #     imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     imagen_redimensionada = cv2.resize(imagen_rgb, (256, 256), interpolation=cv2.INTER_AREA)
# #     imagen_tensor = torch.from_numpy(imagen_redimensionada.transpose(2, 0, 1)).float() / 255.0
# #     imagen_normalizada = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imagen_tensor).unsqueeze(0)

# #     # Realizar la segmentación
# #     with torch.no_grad():
# #         salida = modelo_segmentacion(imagen_normalizada)['out'][0]
# #     salida_prediccion = salida.argmax(0)

# #     # Filtrar los libros en la imagen segmentada (aquí se asume que la clase de libros corresponde al índice 15)
# #     mascara_libros = (salida_prediccion == 15).cpu().numpy().astype(np.uint8)
    
# #     # Encontrar los contornos de los libros
# #     contornos, _ = cv2.findContours(mascara_libros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     # Obtener rectángulos que contienen los libros y las máscaras binarias de cada libro
# #     rectangulos_libros = []
# #     mascaras_binarias = []
# #     for contorno in contornos:
# #         rect = cv2.boundingRect(contorno)
# #         rectangulos_libros.append(rect)

# #         mascara_binaria = np.zeros_like(mascara_libros)
# #         cv2.drawContours(mascara_binaria, [contorno], 0, 255, -1)
# #         mascaras_binarias.append(mascara_binaria)

# #     return rectangulos_libros, mascaras_binarias

# def transformar_imagen(imagen):
#     transformacion = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     imagen_transformada = transformacion(imagen).unsqueeze(0)
#     return imagen_transformada

def transformar_imagen(imagen):
    transformaciones = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # normalizar las imágenes con las medias y desviaciones estándar de ImageNet
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # aplicar transformaciones a la imagen
    imagen_transformada = transformaciones(imagen)
    # agregar dimensión adicional para convertir la imagen en un lote de un solo elemento
    imagen_transformada = imagen_transformada.unsqueeze(0)
    # mover la imagen transformada a la GPU si está disponible
    if torch.cuda.is_available():
        imagen_transformada = imagen_transformada.cuda()
    return imagen_transformada

# # def segmentar_libros(modelo, imagen):
# #     etiqueta_libros = 1
# #     entrada = transformar_imagen(imagen)
# #     salida = modelo(entrada)["out"][0]
# #     salida = salida.argmax(0)
# #     libros_segmentados = (salida == etiqueta_libros).byte().cpu().numpy()
    
# #     return libros_segmentados

# def segmentar_libros(modelo, imagen):
#     # transformar la imagen de entrada
#     entrada = transformar_imagen(imagen)
#     # obtener la salida del modelo
#     with torch.no_grad():
#         salida = modelo(entrada)["out"][0]
#     # obtener la etiqueta correspondiente a los libros
#     etiqueta_libros = 15
#     # segmentar los libros
#     libros_segmentados = (salida.argmax(dim=0) == etiqueta_libros).byte().cpu().numpy()
#     return libros_segmentados

def segmentar_libros(modelo, imagen):
    imagen = imagen[:, :, ::-1]
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    entrada = transformar_imagen(imagen)
    salida = modelo(entrada)["out"][0]
    etiqueta_libros = 15
    libros_segmentados = (salida.argmax(0) == etiqueta_libros).byte().cpu().numpy()
    return libros_segmentados


def crear_rastreador_individual(tipo_rastreador):
    tracker_dict = {
        "BOOSTING": cv2.legacy.TrackerBoosting_create(),
        "CSRT": cv2.legacy.TrackerCSRT_create(),
        "KCF": cv2.legacy.TrackerKCF_create(),
        "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create(),
        "MIL": cv2.legacy.TrackerMIL_create(),
        "MOSSE": cv2.legacy.TrackerMOSSE_create(),
        "TLD": cv2.legacy.TrackerTLD_create()
    }
    
    if tipo_rastreador in tracker_dict.keys():
        rastreador = tracker_dict[tipo_rastreador]
    else:
        raise ValueError(f"El tipo de rastreador '{tipo_rastreador}' no es válido.")
    
    return rastreador


def reconocer_titulos(libros_segmentados, frame):
    # Inicializar el reconocedor de OCR
    reader = easyocr.Reader(['en'], gpu=True)

    titulos = []

    # Iterar sobre los rectángulos de los libros
    for rect in libros_segmentados:
        x, y, w, h = rect

        # Extraer el libro de la imagen
        imagen_libro = frame[y:y+h, x:x+w]

        # Realizar reconocimiento OCR
        resultados_ocr = reader.readtext(imagen_libro)

        # Extraer el texto reconocido
        texto = ' '.join([resultado[1] for resultado in resultados_ocr])

        titulos.append(texto)

    return titulos
