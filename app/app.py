from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# Cargar el modelo entrenado desde el archivo pickle
ruta_modelo = os.path.join(os.getcwd(), "app", "modelo_random_forest.pkl")
with open(ruta_modelo, "rb") as model_file:
    clf = pickle.load(model_file)

# Tamaño deseado para las imágenes 
IMG_SIZE = (512, 384)

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Guardar la imagen para verificar si se recibe bien
    #cv2.imwrite("imagen_recibida.jpg", img)
    
    return img

def predecir_imagen(img_base64):
    img = base64_to_image(img_base64)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalización como en predecir_imagen.py
    img_flat = img.flatten().reshape(1, -1)
    prediccion = clf.predict(img_flat)
    return prediccion[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_base64 = data['image']
    categoria = {
        0: "cartón",
        1: "papel",
        2: "plastico"
    }
    prediccion = predecir_imagen(img_base64)
    return jsonify({'categoria': categoria[prediccion]})

if __name__ == '__main__':
    app.run(debug=True)