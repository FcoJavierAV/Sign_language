import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo entrenado
model = load_model('model/mimodelo.h5')

# Crear la interfaz de usuario
st.title("Clasificación de Lenguaje de Señas")
st.write("Sube una imagen para identificar la letra en lenguaje de señas.")

uploaded_file = st.file_uploader("Sube una imagen en escala de grises 28x28", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar la imagen
    pil_im = Image.open('model/mimodelo.h5', 'r')
    im = np.asarray(pil_im.resize((100,100)))
    im = im.reshape(1,100,100,3)

    # Mostrar la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Predicción
    prediction = model.predict(im)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    st.write(f"Predicción: {classes[np.argmax(prediction)]}")
