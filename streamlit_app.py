import os
import json
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rembg import remove

# Load food_data.json
@st.cache_data
def load_food_data():
    try:
        with open('food_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Food data file not found")
        return []

# Load model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("new_final_model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

food_data = load_food_data()
model = load_model()

class_names = [
    "Alavade", "Aluva", "Bada Irugu", "Cutlet", "Haalpiti Aluwa", "Itli", "Kavum",
    "Kiri Appa", "Kiribath", "Kithul Thalapa", "Konda Kewum", "Levariya", "Mun Kewum",
    "Munguli", "Panivalalu", "Pittu", "Roll", "Roti", "Thala Aggala", "Unduvade",
    "Valithalapa", "Wandu"
]

def preprocess_image(img):
    """Remove background and prepare image for prediction."""
    try:
        img_no_bg = remove(img)
        img_no_bg = img_no_bg.convert("RGB")
        img_array = np.array(img_no_bg)
        img_array = tf.keras.preprocessing.image.img_to_array(img_array)
        img_array = tf.image.resize(img_array, (100, 100))
        img_array = tf.expand_dims(img_array, 0)
        return img_array, img_no_bg
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def predict(img):
    try:
        img_array, img_no_bg = preprocess_image(img)
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
        return fig
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

def main():
    st.title("Food Classification App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if model is None:
            st.error("Model not loaded. Please check server configuration.")
            return
            
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Processing..."):
                predicted_class, confidence, food_info, probabilities, img_no_bg = predict(image)
                fig = generate_prediction_image(img_no_bg, predicted_class, confidence)
                
                # Display results
                st.pyplot(fig)
                
                st.subheader("Prediction Details")
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence}%")
                
                if food_info:
                    st.subheader("Food Information")
                    for key, value in food_info.items():
                        st.write(f"**{key}:** {value}")
                
                # Optional: Show probability distribution
                if st.checkbox("Show probability distribution"):
                    st.subheader("Class Probabilities")
                    st.bar_chart(probabilities)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()