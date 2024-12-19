import cv2
import numpy as np
from matplotlib import pyplot as plt

def sift_product_matching(container_image, product_image):
    """
    Perform product matching using SIFT.

    Args:
        container_image (numpy.ndarray): The container image.
        product_image (numpy.ndarray): The product template image.

    Returns:
        tuple: A tuple containing the matched image and the number of matches.
    """
    try:
        # Validate inputs
        if container_image is None or product_image is None:
            raise ValueError("One or both input images are invalid or not loaded properly.")

        # Convert images to grayscale if they are not already
        if len(container_image.shape) == 3:
            container_image = cv2.cvtColor(container_image, cv2.COLOR_BGR2GRAY)
        if len(product_image.shape) == 3:
            product_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect and compute keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(container_image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(product_image, None)

        # Handle cases where no descriptors are found
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Failed to detect features in one or both images.")

        # Match features using FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors2, descriptors1, k=2)

        # Filter matches using Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # Draw matches
        matched_image = cv2.drawMatches(product_image, keypoints2, container_image, keypoints1, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return matched_image, len(good_matches)

    except Exception as e:
        print(f"Error in SIFT product matching: {e}")
        return None, 0


def orb_product_matching(container_image, product_image):
    """
    Perform product matching using ORB.

    Args:
        container_image (numpy.ndarray): The container image.
        product_image (numpy.ndarray): The product template image.

    Returns:
        tuple: A tuple containing the matched image and the number of matches.
    """
    try:
        # Validate inputs
        if container_image is None or product_image is None:
            raise ValueError("One or both input images are invalid or not loaded properly.")

        # Convert images to grayscale if they are not already
        if len(container_image.shape) == 3:
            container_image = cv2.cvtColor(container_image, cv2.COLOR_BGR2GRAY)
        if len(product_image.shape) == 3:
            product_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect and compute keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(container_image, None)
        keypoints2, descriptors2 = orb.detectAndCompute(product_image, None)

        # Handle cases where no descriptors are found
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Failed to detect features in one or both images.")

        # Match features using BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors2, descriptors1)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches
        matched_image = cv2.drawMatches(product_image, keypoints2, container_image, keypoints1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return matched_image, len(matches)

    except Exception as e:
        print(f"Error in ORB product matching: {e}")
        return None, 0


# Example usage
if __name__ == "__main__":
    # Load example images (replace with actual image data)
    container_image = cv2.imread("container.jpg")
    product_image = cv2.imread("product.jpg")

    # Use SIFT
    sift_image, sift_count = sift_product_matching(container_image, product_image)
    if sift_image is not None:
        print(f"Total Matches Found with SIFT: {sift_count}")
        plt.imshow(sift_image)
        plt.title("SIFT Matching")
        plt.axis("off")
        plt.show()

    # Use ORB
    orb_image, orb_count = orb_product_matching(container_image, product_image)
    if orb_image is not None:
        print(f"Total Matches Found with ORB: {orb_count}")
        plt.imshow(orb_image)
        plt.title("ORB Matching")
        plt.axis("off")
        plt.show()
