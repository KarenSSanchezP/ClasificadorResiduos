import pickle
import cv2
import numpy as np
import base64

# Cargar el modelo entrenado desde el archivo pickle
with open("modelo_random_forest.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

# Tamaño deseado para las imágenes 
IMG_SIZE = (512, 384)

# Función para convertir una imagen a base64
def image_to_base64(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalizar valores de píxeles
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

# Función para convertir base64 a array de numpy
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# Función para predecir la categoría de una imagen
def predecir_imagen(image_path):
    img_base64 = image_to_base64(image_path)
    img = base64_to_image(img_base64)
    img_flat = img.flatten().reshape(1, -1)
    prediccion = clf.predict(img_flat)
    return prediccion[0]

# Ruta de la imagen a predecir (ajusta la ruta según tu estructura)
image_path = "./data/cardboard/cardboard16.jpg"

# Obtener la predicción
categoria_predicha = predecir_imagen(image_path)
print(f"La categoría predicha para la imagen es: {categoria_predicha}")