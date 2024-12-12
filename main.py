import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Streamlit App Title
st.title("Box Label OCR App")
st.write("Upload an image of a box, and the app will extract the text from the label.")

# File Upload Section
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display Uploaded Image
    with st.expander('hide/show uploaded_image'):
        st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR using EasyOCR
    st.write("Extracting text...")
    results = reader.readtext(gray)
    
    # Display Extracted Text
    st.subheader("Extracted Text:")
    for result in results:
        st.write(result[1])  # Display the text content
    
    # Option to Display Grayscale Processed Image
    st.subheader("Processed Grayscale Image:")
    with st.expander('hide/show grayscale image'):
        st.image(gray, caption="Grayscale Image", width=300)
    
