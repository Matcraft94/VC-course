# Creado por: Lucy
# Fecha de creación: 30/04/2023

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def load_emotion_model():
    model = models.alexnet(num_classes=8)
    model.eval()
    return model

def predict_emotion(face_image, model):
    face_image = face_image.resize((224, 224))
    face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(face_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()
        emotion_index = output.argmax().item()
        emotion_probability = probabilities[emotion_index] * 100

    return class_names[emotion_index], emotion_probability
