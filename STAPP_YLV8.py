import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Load YOLOv8 model
@st.cache_resource
def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)

model = load_model()

# Function to run YOLO inference and count products
def detect_products(image):
    # Convert PIL image to OpenCV format
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Perform inference
    results = model(opencv_image)

    # Count bounding boxes
    detected_objects = results[0].boxes
    product_count = len(detected_objects)

    # Draw bounding boxes
    annotated_image = results[0].plot()  # Plot bounding boxes on image
    return annotated_image, product_count

# Streamlit App
st.title("Product Counting with YOLOv8")
st.write("Upload an image of the carton with products to count the items using YOLOv8.")

# Image Upload Section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection
    with st.spinner("Detecting products..."):
        annotated_image, product_count = detect_products(image)

    # Display results
    st.success(f"Total Products Detected: {product_count}")
    st.image(annotated_image, caption="Detected Products", use_column_width=True)

# Footer
st.write("Model: YOLOv8 (Ultralytics) | Streamlit App by YourName")
