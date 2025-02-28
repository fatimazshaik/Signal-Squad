# IMPORTS
import pandas as pd
import math
import numpy as np
from scipy.signal import find_peaks

# FUNCTIONS
# reformat_bb() - reformats bb to be in [x_min, x_max, y_min, y_max] format
def reformat_bb(x_min, y_min, width, height):
    x_max = int(x_min + width)
    y_max = int(y_min + height)
    return [int(x_min), int(y_min), x_max, y_max]

# calculate_IOU() - Calculates the IOU between two boxes
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
        print("Warning: Vertical body detected, skipping linear regression")
        is_horizontal = False
    else:
        try:
            m, _ = np.polyfit(key_x, key_y, 1)  # Fit y = mx + b
            is_horizontal = abs(m) < 0.3  # Threshold for horizontal alignment
        except np.linalg.LinAlgError:
            print("Warning: np.polyfit failed due to singular matrix")
            is_horizontal = False

    # **Keypoint Distance Check**
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_knee_y = (left_knee[1] + right_knee[1]) / 2
    avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2

    shoulder_hip_diff = abs(avg_shoulder_y - avg_hip_y)
    hip_knee_diff = abs(avg_hip_y - avg_knee_y)
    hip_ankle_diff = abs(avg_hip_y - avg_ankle_y)

    is_flat = shoulder_hip_diff < 50 and hip_knee_diff < 50 and hip_ankle_diff < 50

    # Return True if either method detects lying down
    return is_horizontal and is_flat




# IMPORTANT VARIAABLES

# MAIN CODE
object_csv_file = "object_data_bed2.csv"
pose_csv_file = "pose_data_bed2.csv"
count_intersection = 0
interest_frames= []
ious = []
# Load the CSV file without headers and assign custom column names
# Replace 'your_file.csv' with the actual path to your CSV file
df_object = pd.read_csv(object_csv_file, header=None, names=['Frame', 'Object', 'x_center', 'y_center', 'width', 'height'])
df_pose = pd.read_csv(pose_csv_file, header=None, names=['Frame', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear','Left Shoulder','Right Shoulder','Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist','Left Hip','Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle','Right Ankle'])

# Get max frame
frames = df_pose.Frame

# Iterate through all frames:
for i in frames:
    row_index = df_pose.query("Frame == " + str(i)).index.tolist()
    right_shoulder = eval(df_pose.iloc[row_index]["Right Shoulder"].tolist()[0])
    left_shoulder = eval(df_pose.iloc[row_index]["Left Shoulder"].tolist()[0])

    right_hip = eval(df_pose.iloc[row_index]["Right Hip"].tolist()[0])
    left_hip = eval(df_pose.iloc[row_index]["Left Hip"].tolist()[0])
        
    left_knee = eval(df_pose.iloc[row_index]["Left Knee"].tolist()[0])
    right_knee = eval(df_pose.iloc[row_index]["Right Knee"].tolist()[0])

    left_ankle = eval(df_pose.iloc[row_index]["Left Ankle"].tolist()[0])
    right_ankle = eval(df_pose.iloc[row_index]["Right Ankle"].tolist()[0])
    # print(type(left_ankle))
    person_lying_down = 0
    person_lying_down = is_person_lying_down(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle)
    # print("left_shoulder", left_shoulder)

    if(person_lying_down): #og code: iou<0.55 and iou>0.53
        interest_frames.append(i)
        count_intersection += 1

print("data")
print("count_intersection: ", count_intersection)
# print("duration_of_intersection: ", count_intersection/30)
print("interest_frames: ", interest_frames)
# print("START: ", min(interest_frames))
# print("END: ", max(interest_frames))
# print((max(interest_frames) - min(interest_frames))/30)


# for x in interest_frames:
#     row_index = df_pose.query("Frame == " + str(x)).index.to_list()[0]
#     right_shoulder = df_pose.iloc[row_index]["Right Shoulder"]
#     right_elbow = df_pose.iloc[row_index]["Right Elbow"]
#     right_wrist = df_pose.iloc[row_index]["Right Wrist"]
#     left_shoulder = df_pose.iloc[row_index]["Left Shoulder"]
#     left_elbow = df_pose.iloc[row_index]["Left Elbow"]
#     left_wrist = df_pose.iloc[row_index]["Left Wrist"]
#     angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
#     print("Frame: ", x,  "Arm Angle: ", angle)



# print(df[df.Frame==220])
# print(len(df[df.Frame==220]))
# for i in peaks:
#     interest_frames.append(i)
#     # print(ious[i])
# peaks, _ = find_peaks(ious, threshold=0.01)
# print(peaks)