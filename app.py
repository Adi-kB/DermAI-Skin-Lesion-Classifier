import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="DermAI Classifier")

@st.cache_resource
def load_model():
    # Make sure this filename matches exactly what you upload!
    return tf.keras.models.load_model('ham10000_model.h5')

model = load_model()
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

st.title("🩺 Skin Lesion Analyzer")
st.write("Upload an image to classify the lesion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', width=300)
    
    if st.button("Predict"):
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)
        label = class_names[np.argmax(preds)]
        conf = np.max(preds)
        
        st.success(f"Result: {label.upper()} ({conf*100:.2f}%)")