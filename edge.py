import cv2
import numpy as np

# Load the image
image = cv2.imread("carton_top_view.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Perform morphological closing to connect edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and count contours
box_count = 0
for contour in contours:
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter by size (adjust thresholds based on your boxes)
    if 50 < w < 200 and 50 < h < 200:  # Example thresholds
        aspect_ratio = w / h
        if 0.8 < aspect_ratio < 1.2:  # Check for square-like shape
            box_count += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display results
print(f"Number of boxes detected: {box_count}")
cv2.imshow("Detected Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
