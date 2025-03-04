import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import pygame
import datetime
import time
import threading
# import face_recognition
import os

# Initialize pygame for alarm sound
pygame.mixer.init()
ALARM_SOUND = "alarm.wav"  # Ensure you have an alarm sound file
ALARM_DURATION = 3  # Play alarm for 3 seconds
last_alarm_time = 0

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load known faces
# def load_known_faces():
#     known_face_encodings = []
#     known_face_names = []
#     face_dir = "known_faces"
#     if not os.path.exists(face_dir):
#         os.makedirs(face_dir)
    
#     for filename in os.listdir(face_dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(face_dir, filename)
#             image = face_recognition.load_image_file(image_path)
#             encoding = face_recognition.face_encodings(image)
#             if encoding:
#                 known_face_encodings.append(encoding[0])
#                 known_face_names.append(os.path.splitext(filename)[0])
#     return known_face_encodings, known_face_names

# known_face_encodings, known_face_names = load_known_faces()

def play_alarm():
    global last_alarm_time
    if time.time() - last_alarm_time > ALARM_DURATION:
        def alarm_thread():
            pygame.mixer.music.load(ALARM_SOUND)
            pygame.mixer.music.play()
        threading.Thread(target=alarm_thread, daemon=True).start()
        last_alarm_time = time.time()

def is_night_time(start_time, end_time):
    now = datetime.datetime.now().time()
    return start_time <= now or now <= end_time

def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def process_frame(frame, night_mode, start_time, end_time, frame_count):
    if frame_count % 3 != 0:  # Process every 3rd frame to improve performance
        return frame
    
    if night_mode and is_night_time(start_time, end_time):
        frame = enhance_low_light(frame)  # Improve low-light visibility
    
    results = model(frame)
    person_detected = False
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.cpu().numpy()
            if score > 0.5 and model.names[int(class_id)] == "person":
                face_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                # face_encodings = face_recognition.face_encodings(face_frame)
                
                # if face_encodings:
                #     matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                #     if True in matches:
                #         matched_idx = matches.index(True)
                #         name = known_face_names[matched_idx]
                #         label = f"Known: {name}"
                #     else:
                #         label = "Unknown Person"
                #         person_detected = True
                # else:
                    label = "Person"
                    person_detected = True
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if person_detected and night_mode:
        play_alarm()
    
    return frame

def main():
    st.title("CCTV Surveillance with YOLOv8 - Night Security System")
    run = st.checkbox("Start CCTV Stream")
    night_mode = st.checkbox("Enable Night Mode")
    
    st.subheader("Select Night Mode Time")
    start_time = st.time_input("Start Time", value=datetime.time(22, 0))  # Default 10:00 PM
    end_time = st.time_input("End Time", value=datetime.time(6, 0))  # Default 6:00 AM
    
    st.subheader("Upload Known Faces")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        filename = os.path.join("known_faces", uploaded_file.name)
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Added {uploaded_file.name} to known faces.")
    
    cap = cv2.VideoCapture(0)  # Use default webcam
    frame_count = 0  # Counter for frame skipping
    
    if run:
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            
            frame = cv2.resize(frame, (640, 480))  # Optimize performance by resizing
            frame = process_frame(frame, night_mode, start_time, end_time, frame_count)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")
            
            frame_count += 1
        
        cap.release()
    
if __name__ == "__main__":
    main()