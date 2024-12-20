import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

def preprocess_image(image):
    """
    Preprocess the input image for feature detection.
    Converts to grayscale and applies Gaussian blur for noise reduction.

    Args:
        image (numpy array): Input image.

    Returns:
        numpy array: Preprocessed grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def extract_features(image, feature_detector="SIFT"):
    """
    Extract keypoints and descriptors from an image using SIFT or ORB.

    Args:
        image (numpy array): Input preprocessed image.
        feature_detector (str): "SIFT" or "ORB".

    Returns:
        keypoints: List of keypoints.
        descriptors: Feature descriptors.
    """
    if feature_detector == "SIFT":
        detector = cv2.SIFT_create()
    elif feature_detector == "ORB":
        detector = cv2.ORB_create()
    else:
        raise ValueError("Invalid feature detector. Use 'SIFT' or 'ORB'.")

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

def cluster_features(descriptors, eps=3, min_samples=10):
    """
    Cluster features using DBSCAN.

    Args:
        descriptors (numpy array): Feature descriptors.
        eps (float): Maximum distance between two samples for DBSCAN.
        min_samples (int): Minimum number of samples in a cluster.

    Returns:
        labels: Cluster labels for each feature.
    """
    # Check for empty descriptors
    if descriptors is None or len(descriptors) == 0:
        raise ValueError("No descriptors found for clustering.")
    
    # Normalize descriptors for clustering
    descriptors = descriptors.astype(np.float32)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(descriptors)
    return clustering.labels_

def count_products_by_features(image, feature_detector="SIFT", eps=3, min_samples=10):
    """
    Count the most frequent feature group in an image.

    Args:
        image (numpy array): Input image.
        feature_detector (str): "SIFT" or "ORB".
        eps (float): DBSCAN epsilon value.
        min_samples (int): DBSCAN minimum samples.

    Returns:
        int: Estimated number of products.
        numpy array: Image with keypoints drawn.
    """
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Extract features
        keypoints, descriptors = extract_features(preprocessed_image, feature_detector)

        if descriptors is None or len(descriptors) == 0:
            print("No descriptors found in the image.")
            return 0, image

        # Cluster features
        labels = cluster_features(descriptors, eps, min_samples)

        # Count occurrences of each cluster
        unique, counts = np.unique(labels, return_counts=True)
        max_cluster_size = max(counts[unique != -1])  # Exclude noise cluster (-1)

        # Draw keypoints on the original image
        output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return max_cluster_size, output_image

    except ValueError as e:
        print(f"Error: {e}")
        return 0, image

def display_result(image, count, title="Estimated Products"):
    """
    Display the image with the estimated product count.

    Args:
        image (numpy array): Image with keypoints drawn.
        count (int): Estimated product count.
        title (str): Title for the displayed image.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{title}: {count}")
    plt.axis("off")
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Load the container image
    container_image = cv2.imread("container.jpg")

    if container_image is None:
        print("Error: Image file not found or invalid.")
    else:
        # Estimate the number of products
        count, output_image = count_products_by_features(
            container_image,
            feature_detector="SIFT",
            eps=3,
            min_samples=10
        )

        # Display the results
        display_result(output_image, count)
