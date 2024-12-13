import cv2 as cv
import numpy as np
import streamlit as st

st.title('Document Scanner')

# Function to rearrange corner points:
def order_points(pts):
    '''Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point will have the smallest sum.
    rect[2] = pts[np.argmax(s)]  # Bottom-right point will have the largest sum.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right point will have the smallest difference.
    rect[3] = pts[np.argmax(diff)]  # Bottom-left will have the largest difference.
    return rect.astype('int').tolist()

# Function to find destination points:
def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)

# The main function to scan the documents:
def scan(img):
    dim_limit = 1920
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv.resize(img, None, fx=resize_scale, fy=resize_scale)

    orig_img = img.copy()
    kernel = np.ones((7, 7), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=2)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(gray, 50, 150)
    canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    if len(page) == 0:
        return orig_img

    for c in page:
        epsilon = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c, epsilon, True)
        if len(corners) == 4:
            break

    corners = order_points(sorted(np.concatenate(corners).tolist()))
    destination_corners = find_dest(corners)

    maxWidth = destination_corners[2][0]  # Width of the detected document
    maxHeight = destination_corners[2][1]  # Height of the detected document

    M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv.warpPerspective(orig_img, M, (maxWidth, maxHeight), flags=cv.INTER_CUBIC)

    kernel_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final = cv.filter2D(final, -1, kernel_sharpening)

    return final

# Streamlit File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Image of Document:", type=["png", "jpg", "jpeg"])
image = None
final = None

if uploaded_file:
    # Read the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

col1, col2 = st.columns(2)

with col1:
    st.title('Input')
    if image is not None:
        st.image(image, channels='BGR', use_column_width=True)

with col2:
    st.title('Scanned')
    if image is not None:
        final = scan(image)
        st.image(final, channels='BGR', use_column_width=True)
