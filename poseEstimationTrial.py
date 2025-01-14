from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official pse detection model

# Predict with the model
results = model.track(source="video/test1-couch.mp4", save=True)  # predict on an image