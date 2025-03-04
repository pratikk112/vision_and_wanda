import cv2
import torch
import numpy as np
import faiss
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN

# Load YOLOv8 model for face detection
face_detector = YOLO("yolov8n-face.pt")  # Ensure you have a YOLOv8 model trained for faces

# Load FaceNet model for face recognition
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize FAISS index for fast face search
face_db = faiss.IndexFlatL2(512)  # 512-dimensional embeddings
known_faces = []  # List to store names of known faces


def get_face_embeddings(face_image):
    """Extract facial embeddings using FaceNet"""
    face_image = cv2.resize(face_image, (160, 160))  # FaceNet input size
    face_image = np.transpose(face_image, (2, 0, 1)) / 255.0  # Normalize
    face_tensor = torch.tensor(face_image, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        embedding = facenet(face_tensor).numpy()
    return embedding


def add_known_face(face_image, name):
    """Store known face embeddings in FAISS index"""
    embedding = get_face_embeddings(face_image)
    face_db.add(embedding)
    known_faces.append(name)


def recognize_face(face_image):
    """Compare detected face embeddings with stored embeddings"""
    if face_db.ntotal == 0:
        return "Unknown"
    embedding = get_face_embeddings(face_image)
    _, idx = face_db.search(embedding, 1)  # Search for the closest match
    return known_faces[idx[0][0]] if idx[0][0] < len(known_faces) else "Unknown"


def process_frame(frame):
    """Detect faces and recognize them"""
    results = face_detector(frame)
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, _, _ = box.cpu().numpy()
            face = frame[int(y1):int(y2), int(x1):int(x2)]
            if face.size == 0:
                continue
            name = recognize_face(face)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, name, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Start webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
