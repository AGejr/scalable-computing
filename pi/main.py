import cv2 
import numpy as np
import torch

# Load the YOLOv5 model
model = torch.hub.load('/yolov5-master', 'custom', path='best.pt', source='local') #change folders if needed 

 
cap = cv2.VideoCapture(0)  #change cam if needed 
if not cap.isOpened():
    print("Cannot open camera")
    exit()
# Main loop
while True:

    ret, frame = cap.read()
    # Detect objects using YOLOv5
    results = model(frame, stream=True)

    # Process the results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result

        # Draw a rectangle around the object
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

        # Draw the bounding box
        cv2.putText(frame, class_id, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)


    # Show the image
    cv2.imshow("Color Image", frame)
    cv2.waitKey(1)

# Release the VideoWriter object
out.release()
