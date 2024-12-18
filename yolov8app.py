
from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title('Basic Object Detection App using Yolo v8')
# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')
def process(image,model):
    
    # Perform object detection
    results = model(image)

    # Class names for COCO dataset (if using a model trained on COCO)
    class_names = model.names

    # Iterate over the results and draw bounding boxes with labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls)  # Convert tensor to int
            confidence = float(box.conf)  # Convert tensor to float
            class_name = class_names[label]  # Get the class name
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put the label, class name, and confidence
            cv2.putText(image, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            with st.expander('show / hide'):
                st.image(image,caption='detected image')

uploaded_file = st.file_uploader('upload a image ',type=['png', 'jpg','jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    with st.expander('show / hide'):
        st.image(image,caption='uploaded image')
    try:
        process(image,model)
    except Exception as e:
        print(f'got error {e}')
        
        
      
