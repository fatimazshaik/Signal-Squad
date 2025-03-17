# IMPORTS
import pandas as pd
import math
import numpy as np
import statistics
import csv

# FUNCTIONS
def reformat_bb(bb):
    x_max = int(bb[0] + bb[3])
    y_max = int(bb[1] + bb[2])
    return [int(bb[0]), int(bb[1]), x_max, y_max]

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

# GLOBAL VARS
object_csv_file = "CV_Input/object_data.csv"
# object_csv_file = "object_data.csv"
pose_csv_file = "CV_Input/pose_data.csv"
# pose_csv_file = "pose_data.csv"
count_intersection = 0
table_true = 0
new_action = 0
interest_frames= []
action_type_all = []
interest_frames_all = []
action_type = []
ious = []
no_person = True
person_index = 0
object_index = 0
couch_flag = 0
bed_flag = 0
chair_flag = 0
refrigerator_flag = 0
person_flag = 0

# MAIN CODE
df_object = pd.read_csv(object_csv_file, header=None, names=['Frame', 'Object', 'x_center', 'y_center', 'width', 'height'])
df_pose = pd.read_csv(pose_csv_file, header=None, names=['Frame', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear','Left Shoulder','Right Shoulder','Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist','Left Hip','Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle','Right Ankle'])

# Get max frame
max_frame_count = max(df_object.Frame)
frames = df_pose.Frame

# Iterate through all frames:
for i in range(1, max_frame_count+1):
    row_indices = df_object.query("Frame == " + str(i)).index.tolist()
    for x in row_indices:
        if(df_object.loc[x,'Object'] == "couch"):
            couch_flag = 1
            couch_bb = [df_object.loc[x,'x_center'], df_object.loc[x,'y_center'], df_object.loc[x,'width'], df_object.loc[x,'height']]

        elif(df_object.loc[x,'Object'] == "bed"):
            bed_flag = 1
            bed_bb = [df_object.loc[x,'x_center'], df_object.loc[x,'y_center'], df_object.loc[x,'width'], df_object.loc[x,'height']]

        elif(df_object.loc[x,'Object'] == "refrigerator"):
            refrigerator_flag = 1
            refrigerator_bb = [df_object.loc[x,'x_center'], df_object.loc[x,'y_center'], df_object.loc[x,'width'], df_object.loc[x,'height']]

        elif(df_object.loc[x,'Object'] == "chair"):
            chair_flag = 1;
            chair_bb = [df_object.loc[x,'x_center'], df_object.loc[x,'y_center'], df_object.loc[x,'width'], df_object.loc[x,'height']]

        elif(df_object.loc[x,'Object'] == "person"):
            person_flag = 1;
            person_bb = [df_object.loc[x,'x_center'], df_object.loc[x,'y_center'], df_object.loc[x,'width'], df_object.loc[x,'height']]
        else:
            person_flag = 0
    if ((len(df_object[df_object.Frame==i])) >= 2 and person_flag):
        new_action = 0
        if(couch_flag):
            bb1 = reformat_bb(couch_bb)
            bb2 = reformat_bb(person_bb)
        elif (bed_flag):
            bb1 = reformat_bb(bed_bb)
            bb2 = reformat_bb(person_bb)
        elif (chair_flag):
            bb1 = reformat_bb(chair_bb)
            bb2 = reformat_bb(person_bb)
        elif (refrigerator_bb):
            bb1 = reformat_bb(refrigerator_bb)
            bb2 = reformat_bb(person_bb)
        else:
            bb1 = [0,0,0,0]
            bb2 = [0,0,0,0]
        iou = calculate_IOU(bb1, bb2)
            
        if(iou>=0.2 and bed_flag):
                interest_frames.append(df_object.loc[row_indices[person_index], 'Frame'])
                action_type.append("Laying on Bed")

        elif((iou>=0.35) and couch_flag): #Check if Sitting On Couch
            interest_frames.append(df_object.loc[row_indices[person_index], 'Frame'])
            action_type.append("Sitting on Couch")

        elif(iou>=0.3 and refrigerator_flag and not bed_flag): #Check if Opening Fridge og 45
            interest_frames.append(df_object.loc[row_indices[person_index], 'Frame'])
            action_type.append("Opening Fridge")

        elif(iou>=0.35 and chair_flag and not couch_flag): #Check if Sitting on Chair Besides Table
            interest_frames.append(df_object.loc[row_indices[person_index], 'Frame'])
            action_type.append("Sitting on Table")
  
    #Else if statement:
    elif (action_type != []) and (i in df_object.index): 
        if(len(df_object[df_object.Frame==i])==1) and (df_object.loc[i,"Object"]!= "person"):
            if new_action == 30: #og 50
                bed_flag = 0
                couch_flag = 0
                refrigerator_flag = 0
                chair_flag = 0
                action_type_all.append(action_type)
                action_type = []
                interest_frames_all.append(interest_frames)
                interest_frames = []
                new_action = 0
            else:
                new_action += 1
    # no person ion frame & if action [] not empty --> append to ALL ACTIONS
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
                action_type.append("Laying on Bed")

    if((i==max_frame_count) and (action_type != [])): # edge case
        action_type_all.append(action_type)
        action_type = []
        interest_frames_all.append(interest_frames)
        interest_frames = []
    


# print(action_type)
# Get max action_type
# mode = statistics.mode(action_type)
count_actions = 0

with open('CV_Output/iou_output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(["Action Type", "Start Frame", "End Frame"])
    for i in range(len(action_type_all)):
        action = action_type_all[i]
        target_frames = interest_frames_all[i]
        mode = statistics.mode(action)
    # print(action)
        if(len(action) > 10):
            if(action[0] == 'Sitting on Couch' and action[len(action)-1] == 'Sitting on Couch'):
                mode = 'Sitting on Couch'
            if(action[0] == 'Laying on Bed' and action[len(action)-1] == 'Laying on Bed'):
                mode = 'Laying on Bed'
        # Get associated max and min frames
            count_actions +=1
            max_frame = max(target_frames)
            min_frame = min(target_frames)
            writer.writerow([mode, min_frame, max_frame])
            print("Action Type", mode)
            print("Start Frame ", max_frame)
            print("End Frame ", min_frame)
            print("Duration: ", max_frame-min_frame)
        print()
print("Number of Actions: ", count_actions)
