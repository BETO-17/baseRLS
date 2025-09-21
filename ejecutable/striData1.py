'''el  codigo en genera lo que hace es tomar datos de la camara y guardarlos en un archivo csv
    y en una imagen, para luego ser usados en el entrenamiento del modelo pero lo hacer letra por
    letra tambien voacales crea una carpeta por letra y guarda las imagenes en esa carpeta'''

import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

# 🔧 Configuración
DATASET_DIR = "dataset"
IMAGE_SIZE = (200, 200)  # tamaño de las imágenes guardadas
os.makedirs(DATASET_DIR, exist_ok=True)

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # subimos un poco la confianza
    min_tracking_confidence=0.7
) as hands:

    print("✅ Sistema iniciado. Presiona una letra (A-Z) para capturar, ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo acceder a la cámara")
            break

        # Voltear y convertir a RGB
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        # Dibujar landmarks si hay manos detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Captura de manos", frame)

        # Captura de teclado
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC para salir
            print("👋 Saliendo...")
            break

        if results.multi_hand_landmarks and 65 <= key <= 90 or 97 <= key <= 122:
            letter = chr(key).upper()

            # Crear archivos y carpetas para esa letra
            csv_file = os.path.join(DATASET_DIR, f"{letter}.csv")
            images_dir = os.path.join(DATASET_DIR, f"{letter}_images")
            os.makedirs(images_dir, exist_ok=True)

            # Guardar landmarks en CSV
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = []
                    for i in range(21):
                        header += [f"x{i}", f"y{i}", f"z{i}"]
                    writer.writerow(header)

                for hand_landmarks in results.multi_hand_landmarks:
                    data = []
                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])
                    writer.writerow(data)

            # Guardar imagen asociada
            count = len(os.listdir(images_dir))
            img_path = os.path.join(images_dir, f"{letter}_{count+1}_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(img_path, cv2.resize(frame, IMAGE_SIZE))

            print(f"[✔] Captura guardada para letra '{letter}' ({count+1})")

cap.release()
cv2.destroyAllWindows()
