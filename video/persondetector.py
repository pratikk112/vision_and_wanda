import cv2 as cv
from ultralytics import YOLO
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load the notification sound
NOTIFICATION_SOUND = "siren.mp3"  # Replace with your audio file path
pygame.mixer.music.load(NOTIFICATION_SOUND)

# Load the YOLOv8 model (pretrained)
MODEL = YOLO("yolov8n.pt")  # Replace with the model you want to use

# Detection parameters
DETECTION_COOLDOWN = 2  # Seconds between notifications
last_notification_time = 0

def play_notification():
    """
    Plays the notification sound if not already playing.
    """
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def detect_person(frame):
    """
    Runs YOLOv8 detection on the frame and checks for persons.

    Args:
        frame: The current video frame.

    Returns:
        frame: The frame annotated with detection boxes and labels.
        person_detected: True if a person is detected, otherwise False.
    """
    person_detected = False
    results = MODEL(frame, stream=True)

    for result in results:
        boxes = result.boxes  # Detection boxes
        for box in boxes:
            # Extract bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]  # Confidence
            cls = int(box.cls[0])  # Class ID
            label = f"{MODEL.names[cls]}: {conf:.2f}"  # Label with confidence

            # Check if the detected object is a person
            if MODEL.names[cls] == "person":
                person_detected = True

            # Draw bounding box and label
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, person_detected

def main():
    """
    Main function to run YOLOv8 detection with video capture and audio notification.
    """
    global last_notification_time

    # Set up video capture
    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Cannot access the camera")
        return

    # Set camera resolution
    capture.set(3, 720)
    capture.set(4, 720)

    try:
        while True:
            # Read a frame from the camera
            isTrue, frame = capture.read()
            if not isTrue:
                print("Failed to grab frame")
                break

            # Detect objects and check for persons
            frame, person_detected = detect_person(frame)

            # Play notification if a person is detected and cool-down has passed
            current_time = time.time()
            if person_detected and (current_time - last_notification_time > DETECTION_COOLDOWN):
                play_notification()
                last_notification_time = current_time

            # Show the annotated frame
            cv.imshow("YOLOv8 Detection", frame)

            # Exit on pressing 'd'
            if cv.waitKey(1) & 0xFF == ord('d'):
                break

    finally:
        # Release resources properly
        capture.release()
        cv.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
