import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Path to the model file
        model_path = os.path.join("model", "model.h5")

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Ensure it is correctly deployed.")

        # Load model
        model = tf.keras.models.load_model(model_path)

        # Preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0) / 255.0  # Normalize image

        # Perform prediction
        output = model.predict(test_image)
        result = np.argmax(output, axis=1)[0]

        # Map predictions to class names
        class_map = {
            0: 'Adenocarcinoma Cancer',
            1: 'Large cell carcinoma',
            2: 'Normal',
            3: 'Squamous cell carcinoma'
        }
        prediction = class_map.get(result, "Unknown")

        return [{"image": prediction}]
