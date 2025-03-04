import csv
import os

base_path = "/home/jsguo/EEC174/Signal-Squad/CV_Actions"
folder = "couch"
number = 6

for n in range(1, number+1):

    # Set the path to the CSV file (adjust with your actual path)
    csv_file = "/home/jsguo/EEC174/Signal-Squad/CV_Actions/output_vector"

    csv_file = os.path.join(csv_file, f"{folder}{n}-person-tracking.csv")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"File {csv_file} does not exist. Skipping.")
        continue

    # Initialize variables
    threshold = 50  # Threshold for detecting a sitting down or standing up event (in pixels)
    action_events = []

    # Initialize lists to store y_positions and frame indexes
    y_positions = []
    frame_indexes = []

    laying_frame = 0
    up_frame = 0
    # Open the CSV file and read the data
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        
        # Iterate through each row in the CSV file
        for row in reader:
            frame_idx = int(row[1])
            y_position = int(row[3])
            
            # Store frame index and y_position
            frame_indexes.append(frame_idx)
            y_positions.append(y_position)

    # Now, iterate through the frames and compare position[i] with position[i-20]
    for i in range(10, len(y_positions)):
        current_y = y_positions[i]
        previous_y = y_positions[i - 10]
        previous_y_up = y_positions[i - 10]
        frame_idx = frame_indexes[i]
        
        # Detect "sitting down" (laying down): Significant decrease in y_position
        if (current_y > previous_y + threshold) and up_frame == 0:
            action_events.append(('Sitting Down', frame_idx, current_y))
            if (frame_idx - laying_frame > 10):
                laying_frame = frame_idx
        
        # Detect "standing up": Significant increase in y_position
        elif current_y < previous_y_up - threshold and laying_frame != 0:
            action_events.append(('Standing Up', frame_idx, current_y))
            #if (frame_idx - up_frame > 20):
            up_frame = frame_idx

    # # Output the action events (Laying Down and Standing Up)
    # for action, frame_idx, y_position in action_events:
    #     print(f"Action: {action} at frame {frame_idx} with Y position {y_position}")

    diff = up_frame - laying_frame
    print(folder, n)
    print("laying_frame", laying_frame, "up_frame", up_frame)
    print("laying_frame sec", laying_frame/30, "up_frame sec", up_frame/30)
    print("Total Frame: ", up_frame-laying_frame, "Time", diff/30)
    print()
