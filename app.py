import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# Title
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9). The model will predict the digit.")

# Load or define and train your model
@st.cache_resource
def load_model():
    # Load or train model here
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # For demo: Load a pretrained model or train from available MNIST if necessary
    from tensorflow.keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    model.fit(X_train, Y_train, epochs=5, verbose=0)
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_np = np.array(image)
    grayscale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayscale, (28, 28))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28))

    # Prediction
    prediction = model.predict(reshaped)
    predicted_label = np.argmax(prediction)

    st.success(f"The Handwritten Digit is recognized as **{predicted_label}**")
