import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Preprocesses the image to enhance feature detection."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    return blurred

def extract_sift_features(image):
    """Extracts SIFT features from the image."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """Matches features between two sets of descriptors."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: 
            good_matches.append(m)
    return good_matches

def find_most_frequent_features(image, keypoints, matches, num_boxes=20):
    """Identifies frequently matched feature locations."""
    feature_counts = {}
    for match in matches:
        x, y = keypoints[match.queryIdx].pt
        x, y = int(x), int(y) 
        if (x, y) in feature_counts:
            feature_counts[(x, y)] += 1
        else:
            feature_counts[(x, y)] = 1

    # Sort features by occurrence count (descending)
    sorted_features = sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)

    most_frequent_features = []
    for feature, count in sorted_features:
        if count >= num_boxes * 0.7:  # At least 70% of boxes should have this feature
            most_frequent_features.append(feature)
    return most_frequent_features

def visualize_features(image, keypoints, most_frequent_features):
    """Visualizes detected features."""
    for x, y in most_frequent_features:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Mark features in red
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    image_path = 'your_image.jpg'
    processed_image = preprocess_image(image_path)

    # Assuming you have a way to segment individual boxes (see note below)
    all_keypoints = []
    all_descriptors = []
    for box_image in segmented_boxes: 
        keypoints, descriptors = extract_sift_features(box_image)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)

    # Example: Match features between the first two boxes
    matches = match_features(all_descriptors[0], all_descriptors[1])

    # Find most frequent features across all boxes
    most_frequent_features = find_most_frequent_features(processed_image.copy(), all_keypoints[0], matches)

    # Visualize
    visualize_features(processed_image, all_keypoints[0], most_frequent_features)

Important Notes and Next Steps

Box Segmentation: This code assumes you've segmented individual boxes. You might need techniques like contour detection or Hough line transform to isolate the boxes.
Feature Selection: Experiment with different feature extraction methods (SIFT, ORB, SURF) to find what works best for your image characteristics.
Threshold Tuning: Adjust thresholds like the 0.7 values in match_features and find_most_frequent_features to control the sensitivity of feature matching.
Performance: For a large number of boxes, consider optimizing feature matching using techniques like bag-of-words models or vocabulary trees.
Real-World Considerations: Factors like lighting variations, occlusions, and perspective changes in your images can impact feature detection.
Remember to replace "your_image.jpg" with your image path. Let me know if you have any specific questions about implementing the box segmentation or need help with fine-tuning the process!
