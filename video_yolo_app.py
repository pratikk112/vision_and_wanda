import cv2
from ultralytics import YOLO
import streamlit as st import tempfile import os

st.title('OT Solution')

def get_video(): uploaded_file = st.file_uploader("Choose a video...", type=["mp4"]) if uploaded_file is not None: # Create a temporary file to store the uploaded video tfile = tempfile.NamedTemporaryFile(delete=False) tfile.write(uploaded_file.read()) return tfile.name return None

def fun(model_path, video_path): model = YOLO(model_path) # Load the YOLO model once cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object for saving the output video
output_path = 'output.mp4'  # You can customize the output path
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create placeholders for Streamlit video and download button
video_placeholder = st.empty()
download_button = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the processed frame in Streamlit
    video_placeholder.image(frame, channels="BGR", use_column_width=True)

cap.release()
out.release()

# Display download button after processing
with open(output_path, 'rb') as f:
    download_button.download_button(label="Download Output Video", data=f, file_name='output.mp4', mime='video/mp4')

# Clean up the temporary video file
os.remove(video_path)

if name == "main": model_path = 'best.pt' # Replace with your YOLOv8 model path video_path = get_video()

if model_path and video_path:
    try:
        fun(model_path, video_path)
    except Exception as e:
        print(f'Error generated: {e}')
