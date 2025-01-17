import cv2
import math
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# methods:
def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = ang + 360 if ang < 0 else ang
    return 360 - ang if ang > 180 else ang\
    
def draw_angles(annotated_frame, keypoints):
    nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_above_elbow = keypoints[7]
    left_below_elbow = keypoints[9]
    right_above_elbow = keypoints[8]
    right_below_elbow = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    if left_ear_seen and not right_ear_seen:
        angle_knee = get_angle(left_hip, left_knee, left_ankle)
        angle_hip = get_angle(left_shoulder, left_hip, left_knee)
        angle_elbow = get_angle(left_shoulder, left_above_elbow, left_below_elbow)
    else:
        angle_knee = get_angle(right_hip, right_knee, right_ankle)
        angle_hip = get_angle(right_shoulder, right_hip, right_knee)
        angle_elbow = get_angle(right_shoulder, right_above_elbow,  right_below_elbow)

    knee_label_coordinates = [int(c) for c in left_knee]
    knee_label_coordinates[0] += 10
    knee_label_coordinates[1] += 10
    cv2.putText(
        annotated_frame,
        f"{int(angle_knee)}",
        knee_label_coordinates,
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 25, 255),
        2
    )

    hip_label_coordinates = [int(c) for c in left_hip]
    hip_label_coordinates[0] += 10
    hip_label_coordinates[1] += 10
    cv2.putText(
        annotated_frame,
        f"{int(angle_hip)}",
        hip_label_coordinates,
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 25, 255),
        2
    )

    elbow_label_coordinates = [int(c) for c in left_above_elbow]
    elbow_label_coordinates[0] += 10
    elbow_label_coordinates[1] += 10
    cv2.putText(
        annotated_frame,
        f"{int(angle_elbow)}",
        elbow_label_coordinates,
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 25, 255),
        2
    )

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official pse detection model
video_path = "video/test2-brushteeth.mp4"
result_path = "video/result.mp4"

# Tracks an entire video with the model
# results = model.track(source="video/test1-couch.mp4", save=True)  # predict on an image

# Access per frame:
# Load Video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
last_frame_time_seconds = int(time.time())
frames_per_last_second = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height)) #change video name
results = None

while cap.isOpened():
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        print(ret)
        break

    # define variables for bounding boxes, confidence level, inference time, max id detected
    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = frame.shape[:2]

    # Grab Results
    results = model(frame)
    
    if not results:
            continue

    result = results[0]
    frames_per_last_second += 1

    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    keypoints = keypoints[0]
    if not keypoints:
        continue

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)

    annotated_frame = annotator.result()

    draw_angles(annotated_frame, keypoints)

    frame_time = int(time.time())
    if frame_time > last_frame_time_seconds:
        last_frame_time_seconds = frame_time
        print("FPS:", frames_per_last_second)
        fps = frames_per_last_second
        frames_per_last_second = 0

    cv2.putText(
        annotated_frame,
        f"FPS: {fps}",
        (10, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 255, 25),
        1
    )
    out.write(annotated_frame)

cap.release()
out.release()