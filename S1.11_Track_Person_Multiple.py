import cv2
import os
import glob
import csv
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")

# Define the class ID for 'person' (usually 0 in YOLO models)
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.80  # 80% confidence threshold

# Input and output directories
input_dir = "/home/jsguo/EEC174/Signal-Squad/CV_Actions/fridge"
output_video_dir = "/home/jsguo/EEC174/Signal-Squad/CV_Actions/output_video"
output_csv_dir = "/home/jsguo/EEC174/Signal-Squad/CV_Actions/output_vector"
os.makedirs(output_video_dir, exist_ok=True)

# Process each video file in the directory
for video_path in glob.glob(os.path.join(input_dir, "*.mov")):
    filename = os.path.basename(video_path).rsplit(".", 1)[0]  # Get file name without extension
    
    # Create video capture object
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(list)
    last_known_positions = {}  # Store last detected positions for each person

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define output paths
    output_video_path = os.path.join(output_video_dir, f"{filename}-person-only.mov")
    output_csv_path = os.path.join(output_csv_dir, f"{filename}-person-tracking.csv")
        
    frame_idx = 0  # Track current frame index
    actual_person_id = None  # Track the first valid person ID
    

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Store the first video dimensions for consistency
        if frame_idx == 0:  # First frame of the video
            first_width = width
            first_height = height

        # Resize the frame to match the initial dimensions if they differ
        frame_resized = cv2.resize(frame, (first_width, first_height))

        # Run YOLO tracking on the frame
        results = model.track(source=frame, persist=True)

        current_frame_track_ids = set()  # Keep track of persons detected in this frame

        if results[0].boxes.id is not None:
            # Get bounding boxes, track IDs, class IDs, and confidence scores
            boxes = results[0].boxes.xywh.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
                if int(class_id) == PERSON_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:
                    if actual_person_id is None:
                        actual_person_id = track_id  # Assign first detected person ID

                    if track_id == actual_person_id:
                        x, y, w, h = map(int, box[:4])
                        
                        # Calculate bounding box coordinates
                        x1, y1 = x - w // 2, y - h // 2
                        x2, y2 = x + w // 2, y + h // 2

                        track_history[track_id].append((frame_idx, x, y))  # Store actual frame index
                        last_known_positions[track_id] = (x, y)  # Update last known position
                        current_frame_track_ids.add(track_id)

        # Fill missing frames with last known positions
        for track_id in last_known_positions:
            if track_id not in current_frame_track_ids and track_id == actual_person_id:
                track_history[track_id].append((frame_idx, *last_known_positions[track_id]))

        # Annotate frame
        annotated_frame = frame.copy()
        for track_id, track in track_history.items():
            if track_id != actual_person_id:
                continue
            
            if len(track) > 1:
                points = np.array([(x, y) for _, x, y in track], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=4)

            if track_id in last_known_positions:
                x, y = last_known_positions[track_id]
                for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confidences):
                    if tid == track_id and int(cid) == PERSON_CLASS_ID:
                        x1, y1 = int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)
                        x2, y2 = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"ID {track_id} | Conf: {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_idx += 1

    # Release resources
    out.release()
    cap.release()

    # Save tracking data to CSV
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Track ID", "Frame Index", "X", "Y"])
        
        if actual_person_id in track_history:
            for frame_idx, x, y in track_history[actual_person_id]:
                writer.writerow([actual_person_id, frame_idx, x, y])

    print(f"Processed: {video_path}\nTracking data saved in CSV format: {output_csv_path}")
