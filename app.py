import tensorflow as tf
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from io import BytesIO
from PIL import Image

model = tf.keras.models.load_model('diabeticretinopathy_model.h5')
class_names = ['Mild_DR', 'No_DR', 'Severe_DR']

app = Flask(__name__)

def predict_image(img):
    img_resized = cv2.resize(img, (224, 224))  
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_3d = np.expand_dims(img_rgb, axis=0)  # Reshape to (1, 224, 224, 3)
    prediction = model.predict(img_3d)[0]  
    predicted_class = class_names[np.argmax(prediction)]  # Get class with highest probability
    return {"predicted_class": predicted_class}

@app.route('/')
def index():
    return render_template('index1.html')  # Ensure index1.html exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        # Decode base64 image
        header, encoded = image_data.split(",", 1)  # Remove base64 header
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes))
        image = np.array(image)

        # Predict
        prediction_result = predict_image(image)
        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
