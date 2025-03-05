import cv2
import torch
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.neighbors import NearestNeighbors

# Load FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Load MTCNN face detector (better than YOLOv8 for face detection)
mtcnn = MTCNN(keep_all=True)

# Storage for known faces
known_faces = []
known_embeddings = []

# Load existing embeddings (if available)
try:
    with open("face_db.pkl", "rb") as f:
        known_faces, known_embeddings = pickle.load(f)
except FileNotFoundError:
    print("No existing face database found. Creating a new one.")

# Initialize NearestNeighbors model
face_db = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
if known_embeddings:
    face_db.fit(np.vstack(known_embeddings))


def get_face_embeddings(face_image):
    """Extract embeddings using FaceNet."""
    face_image = cv2.resize(face_image, (160, 160))
    face_image = np.transpose(face_image, (2, 0, 1)) / 255.0  # Normalize
    face_tensor = torch.tensor(face_image, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        embedding = facenet(face_tensor).numpy()
    return embedding


def add_known_face(face_image, name):
    """Add a new face to the database."""
    global known_faces, known_embeddings, face_db
    embedding = get_face_embeddings(face_image)
    
    known_embeddings.append(embedding)
    known_faces.append(name)

    # Save updated database
    with open("face_db.pkl", "wb") as f:
        pickle.dump((known_faces, known_embeddings), f)

    # Re-fit the NearestNeighbors model
    face_db.fit(np.vstack(known_embeddings))


def recognize_face(face_image):
    """Recognize a detected face."""
    if len(known_faces) == 0:
        return "Unknown"
    
    embedding = get_face_embeddings(face_image)
    distances, indices = face_db.kneighbors(embedding)
    
    return known_faces[indices[0][0]] if distances[0][0] < 0.6 else "Unknown"


def process_frame(frame):
    """Detect faces and recognize them in real-time."""
    faces, _ = mtcnn.detect(frame)

    if faces is not None:
        for (x1, y1, x2, y2) in faces:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            
            name = recognize_face(face)

            # Draw rectangle & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

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
