import cv2
import torch
from matplotlib import pyplot as plt

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start the webcam feed
cap = cv2.VideoCapture('http://192.168.0.113:8000/stream.mjpg')

while True:
    # Read a frame from the webcam feed
    ret, frame = cap.read()

    # Run the YOLOv5 model on the frame
    results = model(frame)

    # Get the detected objects
    objects = results.xyxy[0]

    # Draw bounding boxes around the detected trees
    for obj in objects:
        if obj[5] == 0:  # Class 0 corresponds to trees in the YOLOv5 model
            x1, y1, x2, y2 = obj[:4].tolist()
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Show the frame with bounding boxes
    cv2.imshow('Tree Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
