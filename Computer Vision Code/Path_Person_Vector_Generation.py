import cv2
import csv
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO("yolo11n.pt")

# Define the class ID for 'person' (usually 0 in YOLO models)
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.70  # 80% confidence threshold

# Create video capture object
cap = cv2.VideoCapture("CV_Input/all_4_actions.mp4")

# Store tracking history and last known frame
track_history = defaultdict(list)
last_known_positions = {}  # Store last detected positions for each person
last_seen_frame = {}  # Stores the last frame a person was detected

# List to store detected person data
tracking_data = []

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer
output_path = "CV_Input/all_4_actions_person_only.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0  # Track current frame index
no_detection_frames = 0  # Counter for frames without detection
current_clip_id = 0  # ID to track which clip we're in
current_tracking_id = None  # ID of the person we're currently tracking
consecutive_no_detection = 0  # Count consecutive frames with no detection
is_start_of_video = True  # Flag to track if we're at the start of the video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(source=frame, persist=True)

    detection_made = False  # Track if any detection was made
    highest_conf_id = None
    highest_conf = 0

    if results[0].boxes.id is not None:
        # Get bounding boxes, track IDs, class IDs, and confidence scores
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # Find the person with highest confidence
        for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
            if int(class_id) == PERSON_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:
                if confidence > highest_conf:
                    highest_conf = confidence
                    highest_conf_id = track_id
                    highest_conf_box = box
        
        # If we found a high-confidence person
        if highest_conf_id is not None:
            x, y, w, h = map(int, highest_conf_box[:4])  # Extract (x, y, width, height)
            
            # If this is the first detection or we've had too many no-detection frames
            if current_tracking_id is None or consecutive_no_detection >= 30:  # Changed from 10 to 30
                # We're in a new clip or starting fresh
                current_clip_id += 1
                current_tracking_id = highest_conf_id
                # Clear previous tracking data for transition to new clip
                track_history.clear()
                last_known_positions.clear()
                last_seen_frame.clear()
            
            # Always track the highest confidence person that meets our threshold
            track_history[current_tracking_id].append((x, y))
            last_known_positions[current_tracking_id] = (x, y)
            last_seen_frame[current_tracking_id] = frame_idx
            tracking_data.append([current_clip_id, frame_idx, x, y, highest_conf])
            detection_made = True
            consecutive_no_detection = 0
            is_start_of_video = False  # We've detected a person, so we're past the start
    
    # Handle missing detection
    if not detection_made:
        consecutive_no_detection += 1
        
        if is_start_of_video:
            # Special case for start of video: insert zeros
            tracking_data.append([0, frame_idx, 0, 0, -3])  # -3 indicates start-of-video with no detection
        elif current_tracking_id is not None:
            frames_since_last_seen = frame_idx - last_seen_frame.get(current_tracking_id, frame_idx)
            
            if frames_since_last_seen <= 5:
                # Short break - use last known position
                x, y = last_known_positions[current_tracking_id]
                tracking_data.append([current_clip_id, frame_idx, x, y, -1])  # -1 indicates interpolated
            else:
                # Long break - probably a transition between clips or temporarily missing
                tracking_data.append([current_clip_id, frame_idx, 0, 0, -2])  # -2 indicates missing
    
    # Write the frame to the output video
    out.write(frame)
    frame_idx += 1  # Update frame count

# Release resources
out.release()
cap.release()

# Save tracking data to CSV
csv_output_path = "CV_Output/person_tracking_results.csv"
with open(csv_output_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Clip_ID", "Frame_Index", "X", "Y", "Confidence"])  # Header
    writer.writerows(tracking_data)

print(f"Tracking data saved to {csv_output_path}")
