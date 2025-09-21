'''captura numeros con mediapipe y opencv
- presiona numeros (0-9) para armar la etiqueta'''
import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

# 🔧 Configuración
DATASET_DIR = "dataset_numbers"
IMAGE_SIZE = (200, 200)
os.makedirs(DATASET_DIR, exist_ok=True)

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Buffer para capturar números de varias cifras
input_buffer = ""

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    print("✅ Sistema iniciado.")
    print("➡ Presiona números (0-9) para armar tu etiqueta.")
    print("➡ ENTER para confirmar y guardar.")
    print("➡ ESC para salir.")

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

        # Mostrar en pantalla el número acumulado
        cv2.putText(frame, f"Etiqueta: {input_buffer}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Captura de manos (Números)", frame)

        # Captura de teclado
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC para salir
            print("👋 Saliendo...")
            break

        # Teclas numéricas (0-9)
        if 48 <= key <= 57:  # códigos ASCII del 0–9
            digit = chr(key)
            if len(input_buffer) < 3:  # máximo 3 cifras
                input_buffer += digit
                print(f"[📝] Buffer: {input_buffer}")

        # Retroceso (BORRAR último número con Backspace)
        if key == 8 and len(input_buffer) > 0:
            input_buffer = input_buffer[:-1]
            print(f"[🔙] Buffer: {input_buffer}")

        # ENTER → Guardar dataset con el número ingresado
        if key == 13 and results.multi_hand_landmarks and input_buffer != "":
            number_label = input_buffer
            print(f"[✔] Guardando captura para número: {number_label}")

            # Crear archivos y carpetas para ese número
            csv_file = os.path.join(DATASET_DIR, f"{number_label}.csv")
            images_dir = os.path.join(DATASET_DIR, f"{number_label}_images")
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
            img_path = os.path.join(images_dir,
                                    f"{number_label}_{count+1}_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(img_path, cv2.resize(frame, IMAGE_SIZE))

            print(f"[📸] Imagen guardada en {img_path}")

            # Resetear buffer
            input_buffer = ""

cap.release()
cv2.destroyAllWindows()
