import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Load YOLOv5 model (pre-trained on COCO or custom trained for your products)
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Small model for speed
    return model

# Detect objects in the image and count products
def detect_and_count_products(image, model):
    # Convert image to OpenCV format
    img_cv2 = np.array(image)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    # Perform inference
    results = model(img_cv2)

    # Extract detections
    detections = results.pandas().xyxy[0]  # DataFrame with bounding boxes and class info
    product_count = len(detections)

    # Draw bounding boxes
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv2, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return product_count, img_cv2

# Streamlit App
st.title("Advanced Product Counting Using YOLOv5")
st.write("Upload an image to detect and count products inside the carton.")

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    with st.spinner("Counting products..."):
        product_count, output_image = detect_and_count_products(image, model)

    # Display results
    st.success(f"Total Products Counted: {product_count}")
    st.image(output_image, caption="Detected Products", use_column_width=True)
