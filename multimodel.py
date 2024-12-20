import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
import os
# from PIL import Image
import streamlit as st
# import cv2 as cv


os.environ['GOOGLE_APPLICATION_CREDENTIALS']='ref'
vertexai.init(project=project, location=location)





IMAGE_FILE = "path"
image = Image.load_from_file(IMAGE_FILE)

generative_multimodal_model = GenerativeModel("gemini-1.5-pro")
response = generative_multimodal_model.generate_content(["prompt ", image])

print(response)
