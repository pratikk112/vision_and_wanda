# STEP 1: Clone YOLOv5 Repository and Install Dependencies
!git clone https://github.com/ultralytics/yolov5  # Clone YOLOv5 repo
%cd yolov5
!pip install -r requirements.txt  # Install dependencies
!pip install opencv-python-headless  # Ensure OpenCV is installed

# STEP 2: Import Required Libraries
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# STEP 3: Load YOLOv5 Pretrained Model
# You can choose 'yolov5s', 'yolov5m', 'yolov5l', or 'yolov5x' based on performance and speed
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# STEP 4: Upload an Image
from google.colab import files
uploaded = files.upload()

# Load the uploaded image
image_path = list(uploaded.keys())[0]  # Take the uploaded image name
image = Image.open(image_path)
plt.imshow(image)
plt.title("Uploaded Image")
plt.axis("off")
plt.show()

# STEP 5: Detect Products and Count Them
results = model(image_path)  # Perform inference
detections = results.pandas().xyxy[0]  # Extract bounding boxes as a Pandas DataFrame

# Count the number of detected objects
product_count = len(detections)

# Display Results
print(f"Total Products Detected: {product_count}")
print(detections[['name', 'confidence']])  # Display object names and confidence scores

# STEP 6: Visualize Detected Products with Bounding Boxes
# Plot the output image with bounding boxes
results.show()  # Show image with bounding boxes
plt.title("Detected Products")
