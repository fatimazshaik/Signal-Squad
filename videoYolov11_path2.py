# Note 1/28
# Works for tracking but need to resolve discontinuetly
# Combine all vector -> present on the map

#Import All the Required Libraries
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

#Load the YOLO Model
model = YOLO("yolo11n.pt")

#Create a Video Capture Object
cap = cv2.VideoCapture(r"EEC174/Signal-Squad/video/test1-couch.mp4")

#Store the Track History
track_history  =defaultdict(lambda  : [])

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Create video writer
output_path = r"EEC174/Signal-Squad/output/test1-couch-out.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

all_track_ids = []
#Loop through the Video Frames
while True:
    ret, frame = cap.read()
    if ret:
        #Run YOLO11 tracking on the frame
        results = model.track(source=frame, persist=True)
        if results[0].boxes.id is not None:
            #Get the bounding box coordinates and the track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for track_id in track_ids:
                # If the track_id is not in all_track_ids_set, add it
                if track_id not in all_track_ids:
                    all_track_ids.append(track_id)
            print("all_track_ids", all_track_ids)

            #Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Create a set of current track IDs detected in the current frame
            current_track_ids = set(track_ids)
            print("current_track_ids", current_track_ids)

            # # Iterate over all tracked objects
            for track_id in all_track_ids:
                if track_id not in current_track_ids:  # If the object is not in the current frame
                    if track_history[track_id]:
                        last_position = track_history[track_id][-1]
                        track_history[track_id].append(last_position)  # Append it to the track history
                        points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color = (230,0,0), thickness=4)
                    else:
                        last_position = None  # or some default value



            for box, track_id, confidence in zip(boxes, track_ids, confidences):
                if confidence > 0.8:
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y))) #x, y center point
                    points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color = (230,0,0), thickness=4)
            out.write(annotated_frame)
    else:
        break
out.release()
cap.release()
# cv2.destroyAllWindows()