import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2

def load_model():
    model = joblib.load('model.joblib')
    # with open('random_forest.pkl', 'rb') as file_new:
    #     data = pickle.load(file_new)

    return model

map_dict = {
    0 : "Defective",
    1 : "Non-defective"
}

model = load_model()

def show_predict_page():

    st.title("Pepper Quality Prediction Application")

    st.write("Upload a picture of peper to check its quality")

    uploaded_image = st.file_uploader("Choose an image file", type = "jpg")

    if uploaded_image != None:
        print("Inside Loop")

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (224, 224))

        st.image(opencv_image, channels = "RGB")

        #processed_img = processor(resized)
        #pred = model.predict()
        #print(uploaded_image.shape)

        Generate_pred = st.button("Generate Prediction")
        if Generate_pred:
            #prediction = model.predict().argmax()
            prediction = 0
            st.title(f"Predicted Label for the image is {map_dict[prediction]}")




