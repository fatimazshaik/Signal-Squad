import pandas as pd
import math
import numpy as np
import statistics
import csv

# VARS for Path
csv_file = "final_result_test/All_couch/all_couch_tracking_path.csv" # CHANGE
threshold_fridge = 700  # Threshold for detecting a sitting down or standing up event (in pixels)
threshold_chair = 100
threshold_couch = 50
threshold_bed = 150
action_events_all = []
action_events = []

# Initialize lists to store y_positions and frame indexes
y_positions = []
x_positions = []
frame_indexes = []

laying_frame = 0
up_frame = 0

# S1: Identify Breaks
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    
    # Iterate through each row in the CSV file
    for row in reader:
        frame_idx = int(row[1])
        y_position = int(row[3])
        x_position = int(row[2])
        
        # Store frame index and y_position
        frame_indexes.append(frame_idx)
        y_positions.append(y_position)
        x_positions.append(x_position)

break_frames = []
break_frame_end = []
in_break = False
start_frame = None
threshold_break = 30


for i in range(len(y_positions)):
    if y_positions[i] == 0:
        if not in_break:
            # Start of a new break
            start_frame = frame_indexes[i]
            in_break = True
        continue  # Continue counting zeros
    else:
        if in_break:
            # End of a break, check if it was long enough
            end_frame = frame_indexes[i - 1]  # Last zero frame
            if (end_frame - start_frame) >= threshold_break:
                break_frames.append(start_frame)
                break_frame_end.append(end_frame)
            in_break = False

# Edge case: If the last frames in the data are a break
# if in_break:
#     break_frames.append(start_frame)

# Print detected breaks
print("Breaks detected (start_frame, end_frame):")
for b in break_frames:
    print(b)

print(break_frame_end)


# S2: Get the object information
csv_file_path = "final_result_test/All_couch/all_couch_object.csv"

data = []
with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        frame = int(row[0])
        obj_class = row[2]
        data.append((frame, obj_class))

# Extract only frame numbers from data
frame_numbers = [item[0] for item in data]
object_list = [item[1] for item in data]
print("frame number", frame_numbers[1:30])

# S3: Determine Action
segments = []
prev_break = 0  # Start from the first frame in `data`

for br in break_frames:
    if br == 0:
        br = 1
    if br in frame_numbers:
        next_frame = frame_numbers.index(br)  # Find exact match
        print(br, next_frame)

    segments.append((prev_break, next_frame))
    prev_break = next_frame

# Add last segment up to the final frame
segments.append((prev_break, len(frame_numbers)))

print(segments)

# Analyze each segment
segment_results = []
i = 0
for start, end in segments:
    print("start/end", start, end)
    if start > end:  # Skip invalid segments
        continue

    segment_data = object_list[start:end]
    # print(segment_data)
    
    unique_classes = set(segment_data)
    class_counts = {cls: segment_data.count(cls) for cls in unique_classes}
    print(class_counts)
    
    # Apply classification rules
    action = "Unknown"
    if "bed" in unique_classes and "couch" in unique_classes and class_counts.get("bed", 0) > 20:
        action = "bed action"
    elif "refrigerator" in unique_classes and class_counts.get("refrigerator", 0) > (end-start) // 4 :
        action = "fridge action"
    elif "couch" in unique_classes and class_counts.get("chair", 0) < class_counts.get("couch", 0):
        action = "couch action"
    elif "chair" in unique_classes and class_counts.get("chair", 0) > class_counts.get("couch", 0):
        action = "chair action"

    if (i == 0):
        segment_results.append((0, break_frames[i], action))
    elif (i == len(segments)-1):
        segment_results.append((break_frame_end[i-1], len(y_positions), action))
    else:
        segment_results.append((break_frame_end[i-1], break_frames[i], action))

    i += 1
# Output the results
for i, (start, end, action) in enumerate(segment_results):
    print(f"Segment {i+1}: List {start}-{end} -> {action}")

previous_action_end = 0

bed_action_start = 0
bed_action_end = 0

next_action_start = 0

bed_merge= []
#S4: Handle the couch -> join the two section
for i, (start, end, action) in enumerate(segment_results):
    if action == "bed action":
        if (i == 0):
            bed_action_end = break_frames[i]
            next_action_start = break_frame_end[i]

            bed_end_average_y = np.average(y_positions[bed_action_end-10:bed_action_end])
            bed_end_average_x = np.average(x_positions[bed_action_end-10:bed_action_end])

            next_action_average_y = np.average(y_positions[next_action_start+1:next_action_start+11])
            next_action_average_x = np.average(x_positions[next_action_start+1:next_action_start+11])

            print("bed_end_average", bed_end_average_y, bed_end_average_x)
            print("next_action_average", next_action_average_y, next_action_average_x)

            if (abs(bed_end_average_y-next_action_average_y) < 50 and abs(bed_end_average_x-next_action_average_x)<50):
                bed_merge.append([i, i+1])
                print(bed_merge)
        
        elif (i == (len(break_frames))):
            previous_action_end = break_frames[i-1]
            bed_action_start = break_frame_end[i-1]
            print(previous_action_end, bed_action_start)

            bed_start_average_y = np.average(y_positions[bed_action_start+1:bed_action_start+10])
            bed_start_average_x = np.average(x_positions[bed_action_start+1:bed_action_start+10])

            previous_action_average_y = np.average(y_positions[previous_action_end-10:previous_action_end])
            previous_action_average_x = np.average(x_positions[previous_action_end-10:previous_action_end])

            print("bed_start_average_y", bed_start_average_y, bed_start_average_x)
            print("previous_action_average_y", previous_action_average_y, previous_action_average_x)

            if (abs(bed_start_average_y-previous_action_average_y) < 50 and abs(bed_start_average_x-previous_action_average_x)<50):
                bed_merge.append([i, i+1])
                print(bed_merge)
        else:
            bed_action_end = break_frames[i]
            next_action_start = break_frame_end[i]

            bed_end_average_y = np.average(y_positions[bed_action_end-10:bed_action_end])
            bed_end_average_x = np.average(x_positions[bed_action_end-10:bed_action_end])

            next_action_average_y = np.average(y_positions[next_action_start+1:next_action_start+11])
            next_action_average_x = np.average(x_positions[next_action_start+1:next_action_start+11])

            print("bed_end_average", bed_end_average_y, bed_end_average_x)
            print("next_action_average", next_action_average_y, next_action_average_x)

            if (abs(bed_end_average_y-next_action_average_y) < 50 and abs(bed_end_average_x-next_action_average_x)<50):
                bed_merge.append([i, i+1])
                print(bed_merge)

            previous_action_end = break_frames[i-1]
            bed_action_start = break_frame_end[i-1]
            print(previous_action_end, bed_action_start)

            bed_start_average_y = np.average(y_positions[bed_action_start+1:bed_action_start+10])
            bed_start_average_x = np.average(x_positions[bed_action_start+1:bed_action_start+10])

            previous_action_average_y = np.average(y_positions[previous_action_end-10:previous_action_end])
            previous_action_average_x = np.average(x_positions[previous_action_end-10:previous_action_end])

            print("bed_start_average_y", bed_start_average_y, bed_start_average_x)
            print("previous_action_average_y", previous_action_average_y, previous_action_average_x)

            if (abs(bed_start_average_y-previous_action_average_y) < 50 and abs(bed_start_average_x-previous_action_average_x)<50):
                bed_merge.append([i, i+1])
                print(bed_merge)

for merge_pair in bed_merge:
    start_idx, end_idx = merge_pair[0] - 1, merge_pair[1] - 1
    
    new_start = segment_results[start_idx][0] 
    new_end = segment_results[end_idx][1]
    new_action = 'bed action' 

    segment_results[start_idx] = (new_start, new_end, new_action)
    del segment_results[start_idx + 1:end_idx + 1]

threshold = {
  "bed action": 150,
  "fridge action": 600,
  "couch action": 80,
  "chair action": 40
}

spacing = {
  "bed action": 20,
  "fridge action": 20,
  "couch action": 10,
  "chair action": 20
}

average_range = 5
laying_frame_y = 0
up_frame_y = 0

# CSV
filename_output = "path_vector/path_output_all4.csv"
headers = ["Action Type", "Start Time", "End Time"]
data_output = []


print("")
for i, (start, end, action) in enumerate(segment_results):
    print(f"Segment {i+1}: List {start}-{end} -> {action}")
    y_pos_here = y_positions[start:end]
    x_pos_here = x_positions[start:end]

    filtered_arr = [x for x in y_pos_here if x != 0]

    max_value = max(y_pos_here)
    max_index = y_pos_here.index(max_value)

    min_value = min(filtered_arr)
    min_index = filtered_arr.index(min_value)

    # print(f"Max value: {max_value} at index {max_index/30}")
    # print(f"Min value: {min_value} at index {min_index/30}")
    # print(x_pos_here)
    for j in range (len(y_pos_here)-5):
        current_y = np.average(y_positions[j:j+average_range])
        previous_y = np.average(y_positions[j - spacing[action]: j - spacing[action] + 5 ])
        previous_y_up = y_positions[j - 20]
        frame_idx = frame_indexes[j]
        # print(current_y, previous_y)
        
        # Detect "sitting down" (laying down): Significant decrease in y_position
        if (current_y > previous_y + threshold[action]) and up_frame == 0:
            action_events.append(('Sitting Down', frame_idx, current_y))
            # First is duration
            # Second considtion is to make sure this movement is not just flat
            if (frame_idx - laying_frame > 10):
                laying_frame = frame_idx
                laying_frame_y = current_y
                # print('Sitting Down', current_y, previous_y, frame_idx + start)
        
        # Detect "standing up": Significant increase in y_position
        elif current_y < previous_y_up - threshold[action] and laying_frame != 0:
            action_events.append(('Standing Up', frame_idx, current_y))
            if (frame_idx - up_frame > 15):
                # print("extrac", np.abs(up_frame_y-current_y))
                up_frame = frame_idx
                up_frame_y = current_y
                # print('Standing Up', current_y, previous_y, frame_idx + start)
    

    diff = up_frame - laying_frame
    print("laying_frame", laying_frame + start, "up_frame", up_frame + start)
    print("laying_frame sec", (laying_frame + start)/30, "up_frame sec", (up_frame + start)/30)
    print("Total Frame: ", up_frame-laying_frame, "Time", diff/30)
    print()
    row = [action, laying_frame + start, up_frame + start]
    data_output.append(row)
print(data_output)

with open(filename_output, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers first
    writer.writerows(data_output)  # Write all rows at once

print(f"CSV file '{filename_output}' has been created with {len(data)} rows.")
