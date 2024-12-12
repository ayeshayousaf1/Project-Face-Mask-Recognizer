import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from skimage.feature import hog  # Import the HOG function

# Load models
cnn_model = load_model('cnn_model.h5')

# Define a function for prediction
def predict_with_svm(image):
    # Preprocess the image for SVM (resize and HOG)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (128, 128))
    hog_features = hog(
        resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'
    ).reshape(1, -1)
    return svm_model.predict(hog_features)[0] # type: ignore

def predict_with_cnn(image):
    # Preprocess the image for CNN (resize and normalize)
    resized_img = cv2.resize(image, (128, 128)) / 255.0
    resized_img = np.expand_dims(resized_img, axis=0)
    predictions = cnn_model.predict(resized_img)
    return np.argmax(predictions, axis=1)[0]

# Streamlit interface
st.title("Face Mask Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)


    # Choose model
    model_type = st.selectbox("Choose Model", ["SVM", "CNN"])

    # Predict
    if st.button("Predict"):
        if model_type == "SVM":
            prediction = predict_with_svm(image)
            st.write(f"Prediction (SVM): {'Mask' if prediction == 1 else 'No Mask'}")
        elif model_type == "CNN":
            prediction = predict_with_cnn(image)
            st.write(f"Prediction (CNN): {'Mask' if prediction == 1 else 'No Mask'}")
