import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo entrenado
model = load_model('model/mimodelo.h5')

# Crear la interfaz de usuario
st.title("Clasificación de Lenguaje de Señas")
st.write("Sube una imagen para identificar la letra en lenguaje de señas.")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen en formato RGB de 100x100", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar la imagen correctamente
    image = Image.open(uploaded_file).convert('RGB')  # Convertir a RGB (por si acaso)
    image = image.resize((100, 100))  # Redimensionar a 100x100
    image_array = np.asarray(image) / 255.0  # Normalizar a valores entre 0 y 1
    image_array = image_array.reshape(1, 100, 100, 3)  # Ajustar dimensiones para la red neuronal

    # Mostrar la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Predicción
    prediction = model.predict(image_array)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Ajusta según tus clases reales
    st.write(f"Predicción: {classes[np.argmax(prediction)]}")
