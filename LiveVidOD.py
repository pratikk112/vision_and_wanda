import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with your YOLO model file

# Open the camera
cap = cv2.VideoCapture(0)  # Replace 0 with the camera index or video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)
    
    # Parse the results
    for result in results:
        boxes = result.boxes  # Detected boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box (top-left and bottom-right)
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display the class name and confidence
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with annotations
    cv2.imshow('Camera', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
