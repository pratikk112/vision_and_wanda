import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use `cv2_imshow` in Google Colab; otherwise, use `cv2.imshow`
try:
    from google.colab.patches import cv2_imshow  # For Google Colab
    SHOW_IMAGE = cv2_imshow
except ImportError:
    # For local environments
    def SHOW_IMAGE(image, title="Image"):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def preprocess_image(image_path):
    """Preprocesses the image to enhance feature detection."""
    image = cv2.imread(image_path)
    SHOW_IMAGE(image, "Original Image")  # Show original image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    SHOW_IMAGE(gray, "Grayscale Image")  # Show grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    SHOW_IMAGE(blurred, "Blurred Image")  # Show blurred image
    return image, blurred

def segment_boxes(image):
    """Segments individual boxes from the image."""
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    SHOW_IMAGE(binary, "Binary Threshold Image")  # Show binary threshold image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # Filter small contours
            box = image[y:y+h, x:x+w]
            SHOW_IMAGE(box, "Segmented Box")  # Show each segmented box
            boxes.append((box, (x, y, w, h)))
    return boxes

def extract_sift_features(image):
    """Extracts SIFT features from the image."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    SHOW_IMAGE(keypoint_image, "SIFT Keypoints")  # Show image with keypoints
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """Matches features between two sets of descriptors."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches

def find_most_frequent_features(keypoints, matches):
    """Identifies frequently matched feature locations."""
    feature_counts = {}
    for match in matches:
        x, y = keypoints[match.queryIdx].pt
        x, y = int(x), int(y)
        feature_counts[(x, y)] = feature_counts.get((x, y), 0) + 1

    # Sort features by occurrence count (descending)
    sorted_features = sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)
    return [feature[0] for feature in sorted_features[:5]]  # Top 5 features

def visualize_features(image, frequent_features):
    """Visualizes detected features."""
    for x, y in frequent_features:
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Mark features in red
    SHOW_IMAGE(image, "Most Frequent Features")  # Show final image

# Main Function
if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with your image path
    original_image, processed_image = preprocess_image(image_path)

    # Segment the boxes
    boxes = segment_boxes(processed_image)

    all_keypoints = []
    all_descriptors = []

    # Extract and visualize SIFT features for each box
    for box, _ in boxes:
        keypoints, descriptors = extract_sift_features(box)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)

    # Match features between the first two boxes (as an example)
    if len(all_descriptors) > 1:
        matches = match_features(all_descriptors[0], all_descriptors[1])
        most_frequent_features = find_most_frequent_features(all_keypoints[0], matches)

        # Visualize most frequent features on the original image
        visualize_features(original_image.copy(), most_frequent_features)
    else:
        print("Not enough boxes detected for feature matching.")
