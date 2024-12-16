import streamlit as st
import cv2
import numpy as np
from PIL import Image
# https://pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/import cv2
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Define Non-Max Suppression (NMS) Function
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Max Suppression to filter overlapping bounding boxes.
    :param boxes: List of bounding boxes [x1, y1, x2, y2]
    :param scores: List of confidence scores for each box
    :param iou_threshold: Overlap threshold for filtering
    :return: Filtered bounding boxes and scores
    """
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    filtered_boxes = []
    filtered_scores = []

    while indices:
        current = indices.pop(0)
        filtered_boxes.append(boxes[current])
        filtered_scores.append(scores[current])

        remaining_indices = []
        for i in indices:
            if iou(boxes[current], boxes[i]) < iou_threshold:
                remaining_indices.append(i)
        indices = remaining_indices

    return filtered_boxes, filtered_scores

# Intersection over Union (IoU) Function
def iou(boxA, boxB):
    """
    Compute IoU between two bounding boxes.
    :param boxA: First bounding box [x1, y1, x2, y2]
    :param boxB: Second bounding box [x1, y1, x2, y2]
    :return: IoU value
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# Template Matching Function
def match_product(container_image, product_image, threshold=0.8):
    """
    Perform template matching and apply NMS to avoid duplicate detections.
    :param container_image: Container image (larger image)
    :param product_image: Template image (product)
    :param threshold: Matching threshold
    :return: Image with bounding boxes and total matches
    """
    container_gray = cv2.cvtColor(container_image, cv2.COLOR_BGR2GRAY)
    product_gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(container_gray, product_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    # Store all bounding boxes and scores
    boxes = []
    scores = []

    for pt in zip(*locations[::-1]):
        x1, y1 = pt
        x2, y2 = x1 + product_image.shape[1], y1 + product_image.shape[0]
        boxes.append([x1, y1, x2, y2])
        scores.append(result[y1, x1])

    # Apply Non-Max Suppression
    filtered_boxes, filtered_scores = non_max_suppression(boxes, scores, iou_threshold=0.5)

    # Draw bounding boxes on the container image
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(container_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return container_image, len(filtered_boxes)

# Streamlit App
st.title("Product Matching with Template Matching and NMS")
st.write("Upload a container image and a product image to detect and count matches.")

# Upload Container Image
uploaded_container = st.file_uploader("Upload Container Image", type=["jpg", "jpeg", "png"])
uploaded_product = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])

if uploaded_container and uploaded_product:
    # Convert uploaded files to OpenCV format
    container_image = np.array(Image.open(uploaded_container))
    product_image = np.array(Image.open(uploaded_product))

    st.image(container_image, caption="Container Image", use_column_width=True)
    st.image(product_image, caption="Product Image", width=200)

    # Perform Matching
    st.write("Detecting products...")
    processed_image, total_matches = match_product(container_image, product_image, threshold=0.8)

    # Display Results
    st.image(processed_image, caption="Detected Matches", use_column_width=True)
    st.success(f"Total Matches Found: {total_matches}")

    # Optionally save the processed image
    output_path = "output_matched_image.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
    st.write("Output image saved as:", output_path)
