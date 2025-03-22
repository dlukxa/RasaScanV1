import os
import json
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rembg import remove  # Import rembg for background removal

app = Flask(__name__)
CORS(app)

# Load food_data.json
try:
    with open('food_data.json', 'r') as f:
        food_data = json.load(f)
    print("Food data loaded successfully.")
except FileNotFoundError as e:
    food_data = []
    print(f"Error: {e}. Food data not loaded.")

class_names = [
    "Alavade", "Aluva", "Bada Irugu", "Cutlet", "Haalpiti Aluwa", "Itli", "Kavum",
    "Kiri Appa", "Kiribath", "Kithul Thalapa", "Konda Kewum", "Levariya", "Mun Kewum",
    "Munguli", "Panivalalu", "Pittu", "Roll", "Roti", "Thala Aggala", "Unduvade",
    "Valithalapa", "Wandu"
]

try:
    model = tf.keras.models.load_model("new_final_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

def preprocess_image(img):
    """Remove background using rembg and prepare image for prediction."""
    try:
        # Remove background
        img_no_bg = remove(img)
        # Convert to RGB if needed (rembg outputs RGBA)
        img_no_bg = img_no_bg.convert("RGB")
        # Convert to numpy array for TensorFlow processing
        img_array = np.array(img_no_bg)
        img_array = tf.keras.preprocessing.image.img_to_array(img_array)
        img_array = tf.image.resize(img_array, (100, 100))
        img_array = tf.expand_dims(img_array, 0)
        return img_array, img_no_bg  # Return processed array and image without background
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def predict(img):
    try:
        # Preprocess image with background removal
        img_array, img_no_bg = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * float(np.max(predictions[0])), 2)
        food_info = next((item for item in food_data if item["Food Name"] == predicted_class), None)
        probabilities = {class_names[i]: round(float(predictions[0][i] * 100), 2) 
                         for i in range(len(class_names))}
        
        return predicted_class, confidence, food_info, probabilities, img_no_bg
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def generate_prediction_image(image, predicted_class, confidence):
    try:
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(np.array(image))
        plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        image_path = 'static/predicted_image.png'
        plt.savefig(image_path)
        plt.close(fig)
        return image_path
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded on server'}), 500

        try:
            image = Image.open(file.stream)
            predicted_class, confidence, food_info, probabilities, img_no_bg = predict(image)
            image_path = generate_prediction_image(img_no_bg, predicted_class, confidence)
            
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'food_info': food_info,
                'image_path': image_path,
                'probabilities': probabilities
            })
        except ValueError as ve:
            print(f"ValueError: {ve}")
            return jsonify({'error': f'Invalid image format: {str(ve)}'}), 400
        except Exception as e:
            print(f"Processing error: {e}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=8000)