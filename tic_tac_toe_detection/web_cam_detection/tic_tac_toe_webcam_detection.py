from roboflow import Roboflow
import supervision as sv
import cv2

# Initialize Roboflow model
rf = Roboflow(api_key="vtoVrzD1Yt2VcFiJDul4")
project = rf.workspace().project("tic-tac-toe-robot")
model = project.version(1).model

# Start capturing video from webcam
cap = cv2.VideoCapture('/dev/video8')  # 0 is the ID for the default webcam

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model.predict(frame, confidence=40, overlap=30).json()
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with detection results
    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = sv.LabelAnnotator().annotate(
        scene=annotated_frame,
        detections=detections,
    )

    # Display the annotated frame
    cv2.imshow("Tic-Tac-Toe Detection", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
