import os
import cv2
import numpy as np
import pandas as pd
import base64
from datasets import load_dataset

# Directorio donde están las imágenes (ajusta la ruta según tu estructura)
data_dir = "./data"

# Tamaño deseado para las imágenes
IMG_SIZE = (512, 384)

# Listas para almacenar imágenes y etiquetas
images = []
labels = []

# Recorrer las subcarpetas (asumiendo que cada subcarpeta es una categoría)
for label, category in enumerate(os.listdir(data_dir)):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)  # Redimensionar
                img = img / 255.0  # Normalizar valores de píxeles
                _, buffer = cv2.imencode('.jpg', img)
                img_str = base64.b64encode(buffer).decode('utf-8')
                images.append(img_str)
                labels.append(label)

# Convertir listas a DataFrame de Pandas
df = pd.DataFrame({"image": images, "label": labels})

# Guardar el DataFrame en un archivo CSV
df.to_csv("dataset_vectorizado.csv", index=False)

# Mostrar información del DataFrame
print(df.head())
