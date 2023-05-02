# Creado por: Lucy
# Fecha de creación: 30/04/2023

import cv2
import face_recognition
from api.emotions import *

def process_frame(frame, face_locations, face_landmarks, emotion_model):
    for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        for feature in landmarks.values():
            for (x, y) in feature:
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                
        face_image = frame[top:bottom, left:right]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_image)
        emotion, emotion_probability = predict_emotion(face_pil, emotion_model)
        cv2.putText(frame, f"{emotion} {emotion_probability:.1f}%", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def main():
    emotion_model = load_emotion_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        face_landmarks = face_recognition.face_landmarks(frame, face_locations)
        processed_frame = process_frame(frame, face_locations, face_landmarks, emotion_model)
        
        cv2.imshow('Face and Emotion Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
