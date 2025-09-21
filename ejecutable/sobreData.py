'''el  codigo en genera lo que hace es tomar datos de la camara y guardarlos en un archivo csv
lo que el codigo e guradar siempre le es tomar las coordenadas de la mano y guardarlas en un archivo csv pero 
con la diferencia que cada vez que se presiona una letra se sobreescribe el archivo csv y la imagen
de esa letra, es decir si se presiona la letra A se guarda en el archivo A.csv y en la imagen A.jpg'''

import cv2
import mediapipe as mp
import csv
import os

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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    print("✅ Sistema iniciado. Presiona una letra (A-Z) para capturar, ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo acceder a la cámara")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Captura de manos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("👋 Saliendo...")
            break

        # ✅ Sobrescribir archivo CSV e imagen
        if results.multi_hand_landmarks and (65 <= key <= 90 or 97 <= key <= 122):
            letter = chr(key).upper()

            # Archivo CSV
            csv_file = os.path.join(DATASET_DIR, f"{letter}.csv")
            with open(csv_file, mode="w", newline="") as f:  # 👈 sobreescribir
                writer = csv.writer(f)
                header = [f"{c}{i}" for i in range(21) for c in ["x", "y", "z"]]
                writer.writerow(header)

                for hand_landmarks in results.multi_hand_landmarks:
                    data = [v for lm in hand_landmarks.landmark for v in [lm.x, lm.y, lm.z]]
                    writer.writerow(data)

            # Imagen: siempre sobrescribe con el mismo nombre
            img_path = os.path.join(DATASET_DIR, f"{letter}.jpg")
            cv2.imwrite(img_path, cv2.resize(frame, IMAGE_SIZE))

            print(f"[✔] Captura sobrescrita para letra '{letter}'")

cap.release()
cv2.destroyAllWindows()
