import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from main_face_recognition import add_known_face  # Importing function from your main face recognition file

# Load existing face database
try:
    with open("face_db.pkl", "rb") as f:
        known_faces, known_embeddings = pickle.load(f)
except FileNotFoundError:
    known_faces, known_embeddings = [], []

# Streamlit UI
st.title("Face Database Management")

# Upload an image
uploaded_image = st.file_uploader("Upload an image to add a new face", type=["jpg", "png"])
name = st.text_input("Enter Name for the Face")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Add Face"):
        add_known_face(image_np, name)
        st.success(f"{name} added successfully!")

# Display existing known faces
st.subheader("Known Faces")
if known_faces:
    for i, face_name in enumerate(known_faces):
        st.write(f"{i+1}. {face_name}")
else:
    st.write("No faces in the database.")

# Option to delete a face
if known_faces:
    delete_name = st.selectbox("Select a face to delete", known_faces)
    if st.button("Delete Face"):
        index = known_faces.index(delete_name)
        del known_faces[index]
        del known_embeddings[index]
        with open("face_db.pkl", "wb") as f:
            pickle.dump((known_faces, known_embeddings), f)
        st.success(f"{delete_name} removed successfully!")
