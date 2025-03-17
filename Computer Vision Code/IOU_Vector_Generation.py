# IMPORT STATMENTS
import cv2
import math
import time
import csv
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import subprocess

# GLOBAL VARIABLES:
frame_counter = 0
input_video_path = "CV_Input/all_4_actions.mp4"
output_video_path = input_video_path
model_pose_estimation = YOLO("yolo11n-pose.pt")  
model_object_detection = YOLO("yolo11x.pt")
object_detections_video = "final_result_test/All_couch/all_4_object_detection.mp4"
final_result_path = "final_result_test/All_couch/all_4_final_result.mp4"
object_data_file_path = "CV_Input/object_data.csv"
pose_data_file_path = "CV_Input/pose_data.csv"


# FUNCTIONS

#Create a CSV File to Store Information
def create_csv_file(csv_file_path):
    # Check if the file already exists to avoid overwriting
    if not Path(csv_file_path).exists():
        with open(csv_file_path, mode='w', newline='') as file:
            writer_csv = csv.writer(file)
            print(f"CSV file created at: {csv_file_path}")
    else:
        print(f"CSV file already exists at: {csv_file_path}")

# #Convert mov file to mp4:
# def convert_mov_to_mp4(input_file, output_file):
#     try:
#         subprocess.run([
#             "ffmpeg", "-i", input_file, "-c:v", "libx264", "-preset", "fast",
#             "-crf", "22", "-c:a", "aac", "-b:a", "192k", "-strict", "experimental", output_file
#         ], check=True)
#         print(f"Conversion successful: {output_file}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error during conversion: {e}")

# import ffmpeg

# def convert_mov_to_mp4(input_file, output_file):
#     try:
#         ffmpeg.input(input_file).output(output_file, vcodec='libx264', acodec='aac', strict='experimental').run()
#         print(f"Conversion successful: {output_file}")
#     except ffmpeg.Error as e:
#         print(f"Error during conversion: {e}")

# import imageio_ffmpeg as iio
# import subprocess

# def convert_mov_to_mp4(input_file, output_file):
#     ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()  # Get the bundled ffmpeg binary path
#     try:
#         subprocess.run([
#             ffmpeg_path, "-i", input_file, "-c:v", "libx264", "-preset", "fast",
#             "-crf", "22", "-c:a", "aac", "-b:a", "192k", "-strict", "experimental", output_file
#         ], check=True)
#         print(f"Conversion successful: {output_file}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error during conversion: {e}")


def key_indexes_from_list(key_items, og_list):
    key_indexes = []
    for item in key_items:
        try:
            index = og_list.index(item)
            key_indexes.append(index)
        except ValueError:
            key_indexes.append(-1)
    return key_indexes

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, frame_counter, csv_file_path, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    frame_detections = []  # Collect detections for this frame

    for result in results:
        for box in result.boxes:
            object_name = result.names[int(box.cls[0])]
            bbox_coords = [int(box.xyxy[0][i]) for i in range(4)]
            detection_str = f"{object_name} ({bbox_coords[0]}, {bbox_coords[1]}, {bbox_coords[2]}, {bbox_coords[3]})"
            # Append detections to the existing CSV file
            with open(csv_file_path, mode='a', newline='') as file:
                writer_csv = csv.writer(file)
                writer_csv.writerow([frame_counter, object_name, str(bbox_coords[0]), str(bbox_coords[1]), str(bbox_coords[2]), str(bbox_coords[3])])
            frame_detections.append(detection_str)
            
            # Draw rectangle and label on the image
            cv2.rectangle(img, (bbox_coords[0], bbox_coords[1]),
                          (bbox_coords[2], bbox_coords[3]), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, object_name,
                        (bbox_coords[0], bbox_coords[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

    return img, frame_detections

def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    print(fps)
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

# Object Detections:
def object_detections(output_video_path, object_detections_video, model_object_detection, csv_file_path, key_indexes):
    cap = cv2.VideoCapture(output_video_path)
    writer = create_video_writer(cap, object_detections_video)
    frame_counter = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        result_img, frame_detections = predict_and_detect(
            model_object_detection, img, frame_counter, csv_file_path, classes=key_indexes, conf=0.5)
        frame_counter += 1
        writer.write(result_img)

    writer.release()
    cap.release()

#Pose Estimation Code:
def process_pose_estimation_video(object_detections_video, final_result_path, model_pose_estimation, csv_file_path, frame_counter):
    cap = cv2.VideoCapture(object_detections_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    last_frame_time_seconds = int(time.time())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_result_path, fourcc, fps, (frame_width, frame_height))
    frame_counter = -1;

    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter+=1;
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        

        # Grab Results
        results = model_pose_estimation(frame)
        if not results:
            results = 0
            continue

        result = results[0]
        keypoints = result.keypoints.xy.tolist()
        if not keypoints:
            keypoints = 0
            continue

        keypoints = keypoints[0]
        if not keypoints:
            keypoints = 0
            continue
        data_to_store = []


        with open(csv_file_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Ensure keypoints is a list before iterating
            if isinstance(keypoints, int):  
                formatted_keypoints = ["[0.0, 0.0]"] * 18  # Placeholder for missing keypoints
            else:
                formatted_keypoints = [f"[{x}, {y}]" for x, y in keypoints]

            # Store frame counter and keypoints
            data_to_store = [frame_counter] + formatted_keypoints

            # Write to CSV
            csv_writer.writerow(data_to_store)

        data_to_store = []

        annotator = Annotator(frame)
        annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
        annotated_frame = annotator.result()

        frame_time = int(time.time())
        if frame_time > last_frame_time_seconds:
            last_frame_time_seconds = frame_time
            print("FPS:", frame_counter)

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



# MAIN STUFF 
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

# convert_mov_to_mp4(input_video_path, output_video_path)
# create_csv_file(object_data_file_path)
# object_detections(output_video_path, object_detections_video, model_object_detection, object_data_file_path, key_indexes)
# # Pose Estimation Stuff
# create_csv_file(pose_data_file_path)
# process_pose_estimation_video(output_video_path, final_result_path, model_pose_estimation, pose_data_file_path, frame_counter)

create_csv_file(object_data_file_path)
object_detections(output_video_path, object_detections_video, model_object_detection, object_data_file_path, key_indexes)

create_csv_file(pose_data_file_path)
process_pose_estimation_video(object_detections_video, final_result_path, model_pose_estimation, pose_data_file_path, frame_counter)
