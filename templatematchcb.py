import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

# Upload files (container image and product image)
uploaded = files.upload()  # Upload container.jpg and product.jpg

# Load images
container_image = cv2.imread('container.jpg', cv2.IMREAD_GRAYSCALE)  # Larger image
product_image = cv2.imread('product.jpg', cv2.IMREAD_GRAYSCALE)  # Template image

# Perform template matching
result = cv2.matchTemplate(container_image, product_image, cv2.TM_CCOEFF_NORMED)

# Define threshold for matches
threshold = 0.8
locations = np.where(result >= threshold)

# Draw rectangles on matched regions
for pt in zip(*locations[::-1]):
    cv2.rectangle(container_image, pt, (pt[0] + product_image.shape[1], pt[1] + product_image.shape[0]), (0, 255, 0), 2)

# Display the result using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(container_image, cmap='gray')
plt.title("Matched Results")
plt.axis("off")
plt.show()
