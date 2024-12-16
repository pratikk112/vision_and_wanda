import cv2

# Load images
container_image = cv2.imread('container.jpg', 0)  # Larger image
product_image = cv2.imread('product.jpg', 0)  # Smaller template image

# Perform template matching
result = cv2.matchTemplate(container_image, product_image, cv2.TM_CCOEFF_NORMED)

# Define threshold for match
threshold = 0.8
locations = np.where(result >= threshold)

# Draw rectangles on matched regions
for pt in zip(*locations[::-1]):
    cv2.rectangle(container_image, pt, (pt[0] + product_image.shape[1], pt[1] + product_image.shape[0]), (0, 255, 0), 2)

# Show result
cv2.imshow("Matched Results", container_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
