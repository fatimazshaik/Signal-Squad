# IMPORT STATMENTS
import cv2
import math
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

### METHODS ###
#get numerical locations of items in another list
def key_indexes_from_list(key_items, og_list):
    key_indexes = []
    for item in key_items:
        try:
            index = og_list.index(item)
            key_indexes.append(index)
        except ValueError:
            key_indexes.append(-1)
    return key_indexes

#get model predictions
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

# get predictions, store into a bounding box + detections in list, draw bounding box on video
def predict_and_detect(chosen_model, img, list_detection, frame_counter, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    item_detections = {}
    for result in results:
        for box in result.boxes:
            item_detections[frame_counter] = {"object": result.names[int(box.cls[0])], "0": int(box.xyxy[0][0]), 
                                              "1": int(box.xyxy[0][1]), "2": int(box.xyxy[0][2]), "3": int(box.xyxy[0][3])}
            list_detection.append(item_detections)
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# defining function for creating a writer (for mp4 videos)
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer

#store list into a file
def store_list_to_file(master_list, filename):
    try:
        with open(filename, 'w') as file:
            for item in master_list:
                file.write(f"{item}\n")
        print(f"List successfully stored in {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

#grab list of objects from .txt file:
def get_list_from_file(filename):
    try:
        with open(filename, 'r') as file:
            datalist = [line.strip() for line in file.readlines()]
            return datalist
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

### Important Variables ###
# Object Paths
model_pose_estimation = YOLO("yolo11n-pose.pt")  
model_object_detection = YOLO("yolo11x.pt")
og_video_path = "input_videos/test1-couch.mp4"
middle_result_path = "results/output_videos/object_detction.mp4"
final_result_path = "results/output_videos/pose_estimation_object_detection.mp4" 
keypoints_video_path = "results/data/keypoints_video.txt"
object_detections_path = "results/data/object_detections.txt"

# Other Set Up Variables
frame_counter = 0;
list_detection = []
keypoints_video = []
results = None
key_object_file_path = "keyObjects.txt"
coco80_lables_file_path = "COCO80.txt"
key_objects = get_list_from_file(key_object_file_path)
coco80_labels = get_list_from_file(coco80_lables_file_path)
key_indexes = key_indexes_from_list(key_objects,coco80_labels)

# Object Detection Portion
cap = cv2.VideoCapture(og_video_path)
writer = create_video_writer(cap, middle_result_path)
while True:
    success, img = cap.read()
    if not success:
        break
    result_img, _ = predict_and_detect(model_object_detection, img, list_detection, frame_counter, classes = key_indexes, conf=0.5)
    frame_counter = frame_counter+1;
    writer.write(result_img)
writer.release()
frame_counter = 0;

# Pose Estimation Portion
cap = cv2.VideoCapture(middle_result_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
last_frame_time_seconds = int(time.time())
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(final_result_path, fourcc, fps, (frame_width, frame_height)) #change video name
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        print(ret)
        break
    # Grab Results
    results = model_pose_estimation(frame)
    if not results:
            continue
    
    result = results[0]
   
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    keypoints = keypoints[0]
    if not keypoints:
        continue
    pose_estimation_dict = {}
    pose_estimation_dict[frame_counter] = keypoints
    keypoints_video.append(pose_estimation_dict) #accumulate keypoints from video frame
    
    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated_frame = annotator.result()
    frame_time = int(time.time())
    if frame_time > last_frame_time_seconds:
        last_frame_time_seconds = frame_time
        print("FPS:", frame_counter)
        fps = frame_counter
        frame_counter = 0

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

    frame_counter = frame_counter +1;

cap.release()
out.release()
frame_counter = 0 # resetting counter

#Storing Important Video Information Into FIle
store_list_to_file(keypoints_video, keypoints_video_path)
store_list_to_file(list_detection, object_detections_path)

print("imdone!")

'''Recyclign Bin:

## Angle Drawing/Calculating Function:
# get angle from three diffferent limbs
def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = ang + 360 if ang < 0 else ang
    return 360 - ang if ang > 180 else ang

#draw angles on annotated video
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

    ## Portion in code is pose_estimation before cap release code:
    frames_per_last_second = 0
    frames_per_last_second += 1
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


Done Recycling''' 