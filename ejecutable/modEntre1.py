'''Reconocimiento de letras con TensorFlow/Keras y MediaPipe
Usa modelo hand_model_tf.h5 y el encoder label_encoder.pkl '''

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Cargar modelo y encoder
model = tf.keras.models.load_model("hand_model_tf.h5")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                data = np.asarray(data).reshape(1, -1)

                # Predicción con TensorFlow
                prediction = model.predict(data, verbose=0)
                class_id = np.argmax(prediction)
                letter = encoder.inverse_transform([class_id])[0]

                # Mostrar resultado
                cv2.putText(frame, f"Letra: {letter}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Dibujar landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
