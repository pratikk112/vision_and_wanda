import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os

st.title("Custom YOLOv5 Object Detection App")

# Function to process the video and display detections
def process_and_display_video(input_path, model):
    # Load the input video
    video = cv2.VideoCapture(input_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Create a temporary file for the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    stframe = st.empty()  # Streamlit's placeholder to display frames dynamically

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = model(rgb_frame)
        detected_frame = np.squeeze(results.render())  # Render detections on the frame

        # Write the detected frame to the output video
        out.write(cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR))

        # Display the current frame with detections
        stframe.image(detected_frame, channels="RGB", use_column_width=True)

    video.release()
    out.release()

    return output_path

# Load the custom YOLOv5 model
@st.cache_resource
def load_custom_model(weights_path):
    return torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, force_reload=True)

# Path to your trained model weights
weights_path = "best.pt"
model = load_custom_model(weights_path)

# Step 1: Upload the video file
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Use a temporary file for the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_video.read())
        input_video_path = temp_input.name

    st.video(input_video_path)  # Display the uploaded video

    # Step 2: Process and display detections frame by frame
    st.write("Processing video...")
    output_video_path = process_and_display_video(input_video_path, model)
    st.success("Processing completed!")

    # Step 3: Provide download option for the processed video
    with open(output_video_path, "rb") as processed_file:
        st.download_button(
            label="Download Processed Video",
            data=processed_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Cleanup: Remove temporary files
    os.remove(input_video_path)
    os.remove(output_video_path)
