# IMPORTS
import pandas as pd
import math
import numpy as np
import statistics
import csv

# FUNCTIONS
def reformat_bb(x_min, y_min, width, height):
    x_max = int(x_min + width)
    y_max = int(y_min + height)
    return [int(x_min), int(y_min), x_max, y_max]

def calculate_IOU(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Calculate the area of the intersection box
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def calculate_angle(point1: str, point2: str, point3: str) -> float:
    # Convert string tuples to actual tuples
    x1, y1 = eval(point1)
    x2, y2 = eval(point2)
    x3, y3 = eval(point3)
    
    # Calculate vectors
    vector_a = (x1 - x2, y1 - y2)  # Vector from point2 to point1
    vector_b = (x3 - x2, y3 - y2)  # Vector from point2 to point3

    # Compute dot product
    dot_product = (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1])

    # Compute magnitudes
    mag_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2)
    mag_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2)

    # Avoid division by zero
    if mag_a == 0 or mag_b == 0:
        return None  # Undefined angle

    # Compute cosine of the angle
    cos_theta = dot_product / (mag_a * mag_b)

    # Clip value to avoid numerical errors (ensuring valid input for arccos)
    cos_theta = max(-1, min(1, cos_theta))

    # Compute angle in degrees
    angle = math.degrees(math.acos(cos_theta))
    
    return angle

def calculate_angle(point1: tuple, point2: tuple, point3: tuple) -> float:
    # Extract coordinates
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Calculate vectors
    vector_a = (x1 - x2, y1 - y2)  # Vector from point2 to point1
    vector_b = (x3 - x2, y3 - y2)  # Vector from point2 to point3

    # Compute dot product
    dot_product = (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1])

    # Compute magnitudes
    mag_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2)
    mag_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2)

    # Avoid division by zero
    if mag_a == 0 or mag_b == 0:
        return 0  # Undefined angle, return 0

    # Compute cosine of the angle
    cos_theta = dot_product / (mag_a * mag_b)

    # Clip value to avoid numerical errors (ensuring valid input for arccos)
    cos_theta = max(-1, min(1, cos_theta))

    # Compute angle in degrees
    return math.degrees(math.acos(cos_theta))

def is_person_lying_down(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee,  left_ankle, right_ankle):
    # Gather x and y coordinates
    key_x = np.array([left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0], left_knee[0], right_knee[0]])
    key_y = np.array([left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1], left_knee[1], right_knee[1]])

    # Handle cases where all x-values are the same (vertical body)
    if np.all(key_x == key_x[0]):
        # print("Warning: Vertical body detected, skipping linear regression")
        is_horizontal = False
    else:
        try:
            m, _ = np.polyfit(key_x, key_y, 1)  # Fit y = mx + b
            is_horizontal = abs(m) < 0.3  # Threshold for horizontal alignment
        except np.linalg.LinAlgError:
            # print("Warning: np.polyfit failed due to singular matrix")
            is_horizontal = False

    # **Keypoint Distance Check**
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_knee_y = (left_knee[1] + right_knee[1]) / 2
    avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2

    shoulder_hip_diff = abs(avg_shoulder_y - avg_hip_y)
    hip_knee_diff = abs(avg_hip_y - avg_knee_y)
    hip_ankle_diff = abs(avg_hip_y - avg_ankle_y)

    is_flat = shoulder_hip_diff < 80 and hip_knee_diff < 80 and hip_ankle_diff < 80

    # Return True if either method detects lying down
    return is_horizontal and is_flat

# # Define input action and number
# folder = "bed"
# number = 7

# GLOBAL VARS
object_csv_file = "object_data.csv"
pose_csv_file = "pose_data.csv"
count_intersection = 0
table_true = 0
new_action = 0
action_type_all = []
interest_frames_all = []
action_type = []
interest_frames= []
ious = []
laying_frame_all = []
up_frame_all = []
no_person = True
person_index = 0
object_index = 0
couch_flag = 0

# VARS for Path
csv_file = "tracking.csv" # CHANGE
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

# MAIN CODE
df_object = pd.read_csv(object_csv_file, header=None, names=['Frame', 'Object', 'x_center', 'y_center', 'width', 'height'])
df_pose = pd.read_csv(pose_csv_file, header=None, names=['Frame', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear','Left Shoulder','Right Shoulder','Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist','Left Hip','Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle','Right Ankle'])

# Get max frame
max_frame_count = max(df_object.Frame)
frames = df_pose.Frame

# Iterate through all frames:
for i in range(1, max_frame_count+1):
    if ((len(df_object[df_object.Frame==i])) >= 2):
        row_indices = df_object.query("Frame == " + str(i)).index.tolist()
        #check if person
        if(df_object.loc[row_indices[0],'Object'] == "person"):
            no_person = False
            person_index = 0
            object_index = 1
        elif (df_object.loc[row_indices[1],'Object'] == "person"):
            no_person = False
            person_index = 1
            object_index = 0
        else:
            no_person = True
        
        if(not no_person):
            new_action = 0
            x_center_bb1 = df_object.loc[row_indices[0], 'x_center']
            x_center_bb2 = df_object.loc[row_indices[1], 'x_center']

            y_center_bb1 = df_object.loc[row_indices[0], 'y_center']
            y_center_bb2 = df_object.loc[row_indices[1], 'y_center']

            width_bb1 = df_object.loc[row_indices[0], 'width']
            width_bb2 = df_object.loc[row_indices[1], 'width']

            height_bb1 = df_object.loc[row_indices[0], 'height']
            height_bb2 = df_object.loc[row_indices[1], 'height']

            bb1 = reformat_bb(x_center_bb1, y_center_bb1, width_bb1, height_bb1)
            bb2 = reformat_bb(x_center_bb2, y_center_bb2, width_bb2, height_bb2)
            iou = calculate_IOU(bb1, bb2)

            # Sitting on the chair
            if (df_object.loc[row_indices[object_index], 'Object'] == "chair"):
                if (iou > 0.3):
                    interest_frames.append(df_object.loc[row_indices[person_index], 'Frame'])
                    action_type.append("Sitting on Table")
                    count_intersection += 1

                    if (i > 30 and i < len(y_positions)):
                        current_y = y_positions[i]
                        previous_y = y_positions[i - 30]
                        previous_y_up = y_positions[i - 30]
                        frame_idx = frame_indexes[i]
                        
                        # Detect "sitting down" (laying down): Significant decrease in y_position
                        if (current_y > previous_y + threshold_chair) and up_frame == 0:
                            action_events.append(('Sitting Down', frame_idx, current_y))
                            if (frame_idx - laying_frame > 10):
                                laying_frame = frame_idx
                        
                        # Detect "standing up": Significant increase in y_position
                        elif current_y < previous_y_up - threshold_chair and laying_frame != 0:
                            action_events.append(('Standing Up', frame_idx, current_y))
                            #if (frame_idx - up_frame > 20):
                            up_frame = frame_idx

            # Opening the fridge
            elif (df_object.loc[row_indices[object_index], 'Object'] == "refrigerator"):
                if (iou>=0.45): 
                    interest_frames.append(df_object.loc[row_indices[person_index], 'Frame'])
                    action_type.append("Opening Fridge")
                    count_intersection += 1

                if (i > 20 and i < len(y_positions)):
                    current_y = y_positions[i]
                    previous_y = x_positions[i - 20]
                    previous_y_up = x_positions[i - 20]
                    frame_idx = frame_indexes[i]

                    # print(i, current_y, previous_y)

                    # Detect "sitting down" (laying down): Significant decrease in y_position
                    if (current_y < previous_y - threshold_fridge) and up_frame == 0:
                        action_events.append(('Sitting Down', frame_idx, current_y))
                        if (frame_idx - laying_frame > 10):
                            laying_frame = frame_idx

                    
                    # Detect "standing up": Significant increase in y_position
                    elif current_y > previous_y_up + threshold_fridge and laying_frame != 0:
                        action_events.append(('Standing Up', frame_idx, current_y))
                        #if (frame_idx - up_frame > 20):
                        up_frame = frame_idx

            # Sitting on couch
            elif (df_object.loc[row_indices[object_index], 'Object'] == "couch"):
                if (iou>=0.4):
                    interest_frames.append(df_object.loc[row_indices[person_index], 'Frame'])
                    action_type.append("Sitting on Couch")
                    count_intersection += 1

                    if (i > 30 and i < len(y_positions)):
                        current_y = y_positions[i]
                        previous_y = y_positions[i - 10]
                        previous_y_up = y_positions[i - 10]
                        frame_idx = frame_indexes[i]
                        
                        # Detect "sitting down" (laying down): Significant decrease in y_position
                        if (current_y > previous_y + threshold_couch) and up_frame == 0:
                            action_events.append(('Sitting Down', frame_idx, current_y))
                            if (frame_idx - laying_frame > 10):
                                laying_frame = frame_idx
                        
                        # Detect "standing up": Significant increase in y_position
                        elif current_y < previous_y_up - threshold_couch and laying_frame != 0:
                            action_events.append(('Standing Up', frame_idx, current_y))
                            #if (frame_idx - up_frame > 20):
                            up_frame = frame_idx
    
    elif (action_type != []) and (i in df_object.index): 
        if(len(df_object[df_object.Frame==i])==1) and (df_object.loc[i,"Object"]!= "person"):
            if new_action == 30:#og 50
                couch_flag = 0
                action_events_all.append(action_events)
                action_events = []
                action_type_all.append(action_type)
                action_type = []
                interest_frames_all.append(interest_frames)
                interest_frames = []
                laying_frame_all.append(laying_frame)
                laying_frame = 0
                up_frame_all.append(up_frame)
                up_frame = 0
                new_action = 0
            else:
                new_action += 1
    else:
        new_action = 0
        row_index = df_pose.query("Frame == " + str(i)).index.tolist()
        if(row_index != []):
            right_shoulder = eval(df_pose.iloc[row_index]["Right Shoulder"].tolist()[0])
            left_shoulder = eval(df_pose.iloc[row_index]["Left Shoulder"].tolist()[0])

            right_hip = eval(df_pose.iloc[row_index]["Right Hip"].tolist()[0])
            left_hip = eval(df_pose.iloc[row_index]["Left Hip"].tolist()[0])
            
            left_knee = eval(df_pose.iloc[row_index]["Left Knee"].tolist()[0])
            right_knee = eval(df_pose.iloc[row_index]["Right Knee"].tolist()[0])

            left_ankle = eval(df_pose.iloc[row_index]["Left Ankle"].tolist()[0])
            right_ankle = eval(df_pose.iloc[row_index]["Right Ankle"].tolist()[0])

            person_lying_down = 0
            person_lying_down = is_person_lying_down(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle)
            # print("left_shoulder", left_shoulder)

            if(person_lying_down): #og code: iou<0.55 and iou>0.53
                interest_frames.append(i)
                count_intersection += 1
                action_type.append("Laying on A Bed")

        if (i > 30 and i < len(y_positions)):
            current_y = y_positions[i]
            previous_y = y_positions[i - 20]
            previous_y_up = y_positions[i - 20]
            frame_idx = frame_indexes[i]
                
                # Detect "sitting down" (laying down): Significant decrease in y_position
            if (current_y > previous_y + threshold_bed) and up_frame == 0:
                action_events.append(('Sitting Down', frame_idx, current_y))
                if (frame_idx - laying_frame > 10):
                    laying_frame = frame_idx
                
            # Detect "standing up": Significant increase in y_position
            elif current_y < previous_y_up - threshold_bed and laying_frame != 0:
                action_events.append(('Standing Up', frame_idx, current_y))
                #if (frame_idx - up_frame > 20):
                up_frame = frame_idx

    if((i==max_frame_count) and (action_type != [])): # edge case
        action_events_all.append(action_events)
        action_events = []
        action_type_all.append(action_type)
        action_type = []
        interest_frames_all.append(interest_frames)
        interest_frames = []
        laying_frame_all.append(laying_frame)
        laying_frame = 0
        up_frame_all.append(up_frame)
        up_frame = 0
# print(action_type)
# Get max action_type


print("Number of Actions: ", len(action_type_all))
for i in range(len(action_type_all)):
    laying_frame = laying_frame_all[i] if i <= (len(laying_frame_all) - 1) else 0
    up_frame = up_frame_all[i] if i <= (len(up_frame_all) - 1) else 0
    event = action_events_all[i] if i <= (len(up_frame_all) - 1) else 0
    action = action_type_all[i]
    target_frames = interest_frames_all[i]
    mode = statistics.mode(action)
    mode1 = statistics.mode(event) if event != [] else 0
    print(mode1)
    print(mode)
    # if(action[0] == 'Sitting on Couch' and action[len(action)-1] == 'Sitting on Couch'):
    #     mode = 'Sitting on Couch'
    # # Get associated max and min frames
    # max_frame = max(target_frames)
    # min_frame = min(target_frames)
    # start_time = min_frame/30
    # end_time = max_frame/30
    # if (abs(start_time - up_frame) < 30):
    #     merge_start_time = (start_time + up_frame)/2
    # else:
    #     if (mode == "Laying on A Bed"):
    #         merge_start_time = up_frame
    #     else: 
    #         merge_start_time = start_time

    # if (abs(end_time - laying_frame) < 30):
    #     merge_end_time = (end_time + laying_frame)/2
    # else:
    #     if (mode == "Laying on A Bed"):
    #         merge_end_time = laying_frame
    #     else: 
    #         merge_end_time = end_time

    # # Print associated start/end time
    # print()
    # print("Action Type: ", mode)
    # print()
    # print("IOU")
    # print("Start Time IOU", start_time)
    # print("End Time IOU", end_time)
    # print("Duration: ", end_time-start_time)
    # print()

    # print("Path")
    # print("Start Time Path", up_frame)
    # print("End Time Path", laying_frame)
    # print("Difference", laying_frame-up_frame)
    # print()

    # print("MERGE TIME")
    # print("Start Time Path", merge_start_time)
    # print("End Time Path", merge_end_time)
    # print("Difference", merge_end_time-merge_start_time)