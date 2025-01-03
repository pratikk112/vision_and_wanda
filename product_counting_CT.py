import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 
# Corrected image path
image_path = 
# Load the image
image = cv2.imread(image_path)
 
# Check if the image is loaded successfully
if image is None:
    print(f"Error: Image not found at {image_path}")
    exit()
 
 
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
 
# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
 
# Find contours in the adaptive threshold image
contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# Filter out contours that are too small (less than a certain area)
min_area = 1000 # Adjust based on the image and box size
max_area = 10000
valid_contours = []
 
# Iterate over contours and apply additional filtering based on aspect ratio and bounding box shape
for cnt in contours:
    # Get the bounding box around the contour
    x, y, w, h = cv2.boundingRect(cnt)
   
    area = cv2.contourArea(cnt)
    # Filter based on minimum area
    if (area < min_area) and (area > max_area):
        continue
   
   
    if (h / float(w)) < 0.2:
        continue
    # Filter based on aspect ratio to exclude QR codes or other similar shapes
    aspect_ratio = float(w) / h
    if 0.6 < aspect_ratio < 1.2:  # Exclude square-shaped objects like QR codes (adjust threshold as needed)
        continue
   
    valid_contours.append(cnt)
 
# Count the number of valid boxes
num_boxes = len(valid_contours)
 
# Create a black image for the filtered edges
filtered_edges = np.zeros_like(adaptive_thresh)
 
# Draw the filtered contours on the black image (only edges considered as boxes)
cv2.drawContours(filtered_edges, valid_contours, -1, 255, 1)  # White color (255) for box edges
 
# Create a copy of the original image to draw contours (for final result)
output_image = image.copy()
 
# Draw the contours on the image
cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 3)  # Green color for contours
 
 
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Boxes: {num_boxes}')
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(filtered_edges, cmap='gray')
plt.title('Filtered Edges (Boxes Only)')
plt.axis('off')
 
plt.show()
