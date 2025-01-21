import streamlit as st
import torch
import tempfile
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import os

st.title("Custom YOLOv5 Object Detection App with MoviePy")

# Function to process video and display detections with frame skipping
def process_and_display_video(input_path, model, skip_frames=5):
    # Load the input video using MoviePy
    video_clip = VideoFileClip(input_path)
    fps = video_clip.fps

    # Extract frames and perform detection
    processed_frames = []
    stframe = st.empty()  # Streamlit placeholder to display frames dynamically

    for i, frame in enumerate(video_clip.iter_frames(fps=fps, dtype="uint8")):
        # Skip frames for faster processing
        if i % skip_frames != 0:
            continue

        # Perform detection
        results = model(frame)
        detected_frame = np.squeeze(results.render())  # Render detections on the frame
        processed_frames.append(detected_frame)

        # Display the current frame
        stframe.image(detected_frame, channels="RGB", use_column_width=True)

    # Clear the frame display when processing is complete
    stframe.empty()

    # Create the output video using MoviePy
    output_clip = ImageSequenceClip(processed_frames, fps=fps // skip_frames)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_clip.write_videofile(temp_output.name, codec="libx264", audio=False)
        output_path = temp_output.name

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
    st.write("Processing video... This might take a while.")
    output_video_path = process_and_display_video(input_video_path, model, skip_frames=5)
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
