# STEP 1: Install Ultralytics (YOLOv8)
!pip install ultralytics
import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# STEP 2: Check GPU Availability
print("PyTorch CUDA Status:", torch.cuda.is_available())

# STEP 3: Load YOLOv8 Pretrained Model
# Model Options: 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 'yolov8m.pt' (medium), 'yolov8l.pt' (large), 'yolov8x.pt' (extra large)
model = YOLO("yolov8n.pt")  # Use nano model for speed (change to larger models for better accuracy)

# STEP 4: Upload an Image
from google.colab import files
uploaded = files.upload()

# Get image path
image_path = list(uploaded.keys())[0]
print(f"Uploaded file: {image_path}")

# STEP 5: Perform Inference (Detection)
results = model(image_path)  # Run detection on the uploaded image

# STEP 6: Display Total Products Counted
detected_objects = results[0].boxes  # Access bounding box results
product_count = len(detected_objects)  # Count total detected bounding boxes
print(f"Total Products Detected: {product_count}")

# STEP 7: Visualize the Detection Results
# Display image with bounding boxes
results[0].show()

# STEP 8: Save Output Image
output_image_path = "output_detected_image.jpg"
results[0].save(filename=output_image_path)
print(f"Output image saved at: {output_image_path}")

# STEP 9: Display the Final Image with Bounding Boxes
output_image = Image.open(output_image_path)
plt.imshow(output_image)
plt.title("Detected Products")
plt.axis("off")
plt.show()
