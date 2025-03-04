import csv
import os

base_path = "/home/jsguo/EEC174/Signal-Squad/CV_Actions"
folder = "bed"
number = 7

for n in range(1, number+1):

    # Set the path to the CSV file (adjust with your actual path)
    csv_file = "/home/jsguo/EEC174/Signal-Squad/CV_Actions/output_vector"

    csv_file = os.path.join(csv_file, f"{folder}{n}-person-tracking.csv")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"File {csv_file} does not exist. Skipping.")
        continue

    # Initialize variables
    threshold = 150  # Threshold for detecting a sitting down or standing up event (in pixels)
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
    for i in range(20, len(y_positions)):
        current_y = y_positions[i]
        previous_y = y_positions[i - 20]
        previous_y_up = y_positions[i - 20]
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

    if up_frame == 0:
        up_frame = i
    diff = up_frame - laying_frame
    print(folder, n)
    print("laying_frame", laying_frame, "up_frame", up_frame)
    print("laying_frame sec", laying_frame/30, "up_frame sec", up_frame/30)
    print("Total Frame: ", up_frame-laying_frame, "Time", diff/30)
    print()


# import csv
# import os

# # Set the base path where all the CSV files are located
# base_path = "/home/jsguo/EEC174/Signal-Squad/CV_Actions/output_vector"
# output = []

# # Initialize variables
# threshold = 150  # Threshold for detecting a sitting down or standing up event (in pixels)

# # Loop through all couch vector files from couch1 to couch7
# for i in range(1, 8):
#     csv_file = os.path.join(base_path, f"couch{i}-person-tracking.csv")
    
#     # Check if the CSV file exists
#     if not os.path.exists(csv_file):
#         print(f"File {csv_file} does not exist. Skipping.")
#         continue

#     # Initialize lists to store y_positions and frame indexes
#     y_positions = []
#     frame_indexes = []
#     action_events = []

#     # Open the CSV file and read the data
#     with open(csv_file, mode='r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip the header row

#         # Iterate through each row in the CSV file
#         for row in reader:
#             frame_idx = int(row[1])
#             y_position = int(row[3])

#             # Store frame index and y_position
#             frame_indexes.append(frame_idx)
#             y_positions.append(y_position)

#     laying_frame = 0
#     up_frame = 0
#     # Now, iterate through the frames and compare position[i] with position[i-20]
#     for i in range(20, len(y_positions)):
#         current_y = y_positions[i]
#         previous_y = y_positions[i - 20]  # y_position from 20 frames ago
#         previous_y_up = y_positions[i - 10]
#         frame_idx = frame_indexes[i]

#         # Detect "sitting down" (laying down): Significant decrease in y_position
#         if current_y > previous_y + threshold:
#             action_events.append(('Laying Down', frame_idx, current_y))
#             if (frame_idx - laying_frame > 20):
#                 laying_frame = frame_idx

#         # Detect "standing up": Significant increase in y_position
#         elif current_y < previous_y_up - threshold:
#             action_events.append(('Standing Up', frame_idx, current_y))
#             #if (frame_idx - up_frame > 20):
#             up_frame = frame_idx

#     diff = up_frame - laying_frame
#     print("laying_frame", laying_frame, "up_frame", up_frame)
#     print("laying_frame sec", laying_frame/30, "up_frame sec", up_frame/30)
#     print("Total Frame: ", up_frame-laying_frame, "Time", diff/30)

#     # # Output the action events for this file
#     # print(f"Processing {csv_file}...")
#     # for action, frame_idx, y_position in action_events:
#     #     print(f"Action: {action} at frame {frame_idx} with Y position {y_position}")
    
#     # # Save output for later if needed (optional)
#     # output.append((csv_file, action_events))

# # Optionally save all action events to a file or process further
# # For example, you could write them to a summary CSV or store them in a dictionary for later use.
