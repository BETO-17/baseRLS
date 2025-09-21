'''Entrenamiento de modelo de reconocimiento de letras con TensorFlow/Keras
   - Dataset: CSVs en carpeta dataset/ (uno por letra) Y crea un modelo hand_model_tf.h5'''

import pandas as pd
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# 1️⃣ Cargar todos los CSV del dataset
all_files = glob.glob("dataset/*.csv")
dataframes = []

for file in all_files:
    df = pd.read_csv(file)
    label = file.split("\\")[-1].split(".")[0]  # Nombre archivo = etiqueta
    df["label"] = label
    dataframes.append(df)

# Unir todo
data = pd.concat(dataframes, ignore_index=True)

# 2️⃣ Preparar datos
X = data.drop("label", axis=1).values
y = data["label"].values

# Codificar etiquetas (A,E,I,O,U → 0,1,2,3,4)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Guardar encoder para usar luego
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# 3️⃣ Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Definir modelo Keras
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),  # capa entrada
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation="softmax")  # salida = nº de clases
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 5️⃣ Entrenar
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# 6️⃣ Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Precisión en test: {acc:.2f}")

# 7️⃣ Guardar modelo
model.save("hand_model_tf.h5")
print("Modelo guardado en hand_model_tf.h5")
