from roboflow import Roboflow
import supervision as sv
import cv2
rf = Roboflow(api_key="vtoVrzD1Yt2VcFiJDul4")
project = rf.workspace().project("tic-tac-toe-robot")
model = project.version(1).model

IMAGE_PATH = "/home/qasob/Tic-Tac-Toe-Playing-Robot/tic_tac_toe_images/5328094608330187829.jpg"
image = cv2.imread(IMAGE_PATH)
# infer on a local image
results = model.predict(image, confidence=40, overlap=30).json()

detections = sv.Detections.from_inference(results)

annotated_image = image.copy()

annotated_image = sv.BoxAnnotator().annotate(
    scene=annotated_image,
    detections=detections,
)

annotated_image = sv.LabelAnnotator().annotate(
    scene=annotated_image,
    detections=detections,
)


sv.plot_image(annotated_image)

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())