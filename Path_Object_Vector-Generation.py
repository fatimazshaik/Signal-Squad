import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import csv

# Load YOLO model
model = YOLO("yolo11n.pt")

# Create video capture object
input_video = "final_result_test/All_couch/ALL_COUCH.MOV"
cap = cv2.VideoCapture(input_video)

# Store the track history
track_history = defaultdict(lambda: [])

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer
output_path = "final_result_test/All_couch/object_result.MOV"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# For CSV output
csv_output = "final_result_test/All_couch/all_couch_object.csv"
csv_file = open(csv_output, 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(['Frame', 'Track_ID', 'Class_Name', 'X', 'Y', 'W', 'H'])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Run YOLO tracking on the frame
    results = model.track(source=frame, persist=True)
    
    # Get the annotated frame with boxes
    annotated_frame = results[0].plot()
    
    if results[0].boxes.id is not None:
        # Get bounding boxes, track IDs, and class IDs
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.cpu().tolist()
        class_names = [model.names[int(c)] for c in class_ids]  # Get class names
        
        # Process each detection
        for box, track_id, class_id, class_name in zip(boxes, track_ids, class_ids, class_names):
            x, y, w, h = map(int, box)  # Convert to integers
            
            # Add detection to track history
            track_history[track_id].append((x, y))
            
            # Draw tracking lines
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=2)
            
            # Add class name and track ID on top of detection box
            label = f"{class_name} ({track_id})"
            cv2.putText(annotated_frame, label, (x - w//2, y - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame data to CSV (Frame, Track_ID, Class_Name, X, Y, W, H)
            csv_writer.writerow([frame_count, track_id, class_name, x, y, w, h])
    
    # Write frame to output video
    out.write(annotated_frame)

# Release all resources
cap.release()
out.release()
csv_file.close()

print(f"Tracking complete. Results saved to {output_path} and {csv_output}")
