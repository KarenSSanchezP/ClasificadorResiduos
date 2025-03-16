import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import base64
import cv2

# Cargar el dataset desde el CSV
df = pd.read_csv("dataset_vectorizado.csv")

# Convertir la columna de imágenes de base64 a arrays de numpy
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

df["image"] = df["image"].apply(base64_to_image)

# Aplanar las imágenes para que sean vectores en lugar de matrices
X = np.array([img.flatten() for img in df["image"]])
y = df["label"].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Guardar el modelo entrenado en un archivo pickle
with open("modelo_random_forest.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

print("Modelo guardado como 'modelo_random_forest.pkl'")
