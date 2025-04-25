import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('road_sign_model.h5')

# Load class labels
class_labels = pd.read_csv('labels.csv')
class_names = class_labels['Name'].tolist()

# Streamlit app title and instructions
st.title('🚦 Road Sign Classifier')
st.write('Upload a road sign image to identify its class.')

# Upload file widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_image(img):
    image = img.resize((160, 160)) 
    img_array = np.array(image) / 160.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

# If file uploaded, display and predict
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        predicted_class, confidence = predict_image(image)
        st.success(f"Prediction: {class_names[predicted_class]}")
        st.info(f"Confidence: {confidence:.2f}")


