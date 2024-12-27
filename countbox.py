import cv2
import numpy as np
from sklearn.cluster import KMeans  # Import KMeans for clustering

# Load the image
image_path = 'carton_top_view.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess the image (e.g., binarization)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours to segment the items
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize SIFT for feature extraction
sift = cv2.SIFT_create()

# List to hold descriptors of each item
all_descriptors = []

# Loop through contours to extract features
segmented_images = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 20 and h > 20:  # Ignore small artifacts
        item = gray[y:y+h, x:x+w]
        segmented_images.append((item, (x, y, w, h)))
        
        # Detect keypoints and descriptors
        _, descriptors = sift.detectAndCompute(item, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)

# Stack all descriptors for clustering
if all_descriptors:  # Ensure there are descriptors to cluster
    all_descriptors = np.vstack(all_descriptors)

    # Use KMeans to cluster descriptors
    k = 5  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(all_descriptors)

    # Find the most frequent cluster
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    most_frequent_cluster = unique[np.argmax(counts)]

    # Highlight items containing the most frequent cluster
    highlighted_image = image.copy()
    for item, (x, y, w, h) in segmented_images:
        _, descriptors = sift.detectAndCompute(item, None)
        if descriptors is not None:
            cluster_labels = kmeans.predict(descriptors)
            if most_frequent_cluster in cluster_labels:
                cv2.rectangle(highlighted_image, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("Highlighted Image", highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No features found in the image.")
