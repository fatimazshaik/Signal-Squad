import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

def process_video(video_path, output_path, conf_threshold=0.25):

    model = YOLO('yolo11n.pt')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=conf_threshold)
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()
        # Draw the detections manually
        # for result in results:
        #     for box in result.boxes:
        #         # Get box coordinates
        #         x1, y1, x2, y2 = map(int, box.xyxy[0])
                
        #         # Get confidence
        #         confidence = float(box.conf[0])
                
        #         # Get class name
        #         class_id = int(box.cls[0])
        #         class_name = model.names[class_id]
                
        #         # Draw bounding box
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        #         # Draw label
        #         label = f'{class_name} {confidence:.2f}'
        #         cv2.putText(frame, label, (x1, y1 - 10), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        
        # Write the frame
        out.write(frame)
        
        # # Display the frame
        # cv2.imshow("YOLOv8 Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    input_video = r"EEC174/Signal-Squad/video/test1-couch.mp4"
    output_video = r"EEC174/Signal-Squad/output/test1-couch.mp4"
    
    # You can customize these parameters
    process_video(
        video_path=input_video,
        output_path=output_video,
        conf_threshold=0.25,  # Confidence threshold (0-1)
    )