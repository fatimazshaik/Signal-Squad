# IMPORTS
import pandas as pd
import math
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

# IMPORTANT VARIAABLES

# MAIN CODE
object_csv_file = "object_data_fridge4.csv"
pose_csv_file = "pose_data_fridge4.csv"
count_intersection = 0
interest_frames= []
ious = []
# Load the CSV file without headers and assign custom column names
# Replace 'your_file.csv' with the actual path to your CSV file
df_object = pd.read_csv(object_csv_file, header=None, names=['Frame', 'Object', 'x_center', 'y_center', 'width', 'height'])
df_pose = pd.read_csv(pose_csv_file, header=None, names=['Frame', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear','Left Shoulder','Right Shoulder','Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist','Left Hip','Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle','Right Ankle'])

# Get max frame
max_frame_count = max(df_object.Frame)

# Iterate through all frames:
for i in range(1, max_frame_count+1):
    if ((len(df_object[df_object.Frame==i])) >= 2):
        # Perform IOU Operation
        row_indices = df_object.query("Frame == " + str(i)).index.tolist()
        #check if person & fridge
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

        if(iou>=0.45): #og code: iou<0.55 and iou>0.53
            interest_frames.append(df_object.loc[row_indices[0], 'Frame'])
            count_intersection += 1

print("data")
print("count_intersection: ", count_intersection)
print("duration_of_intersection: ", count_intersection/30)
# print("interest_frames: ", interest_frames)
print("START: ", min(interest_frames))
print("END: ", max(interest_frames))
print((max(interest_frames) - min(interest_frames))/30)

for x in interest_frames:
    row_index = df_pose.query("Frame == " + str(x)).index.to_list()[0]
    right_shoulder = df_pose.iloc[row_index]["Right Shoulder"]
    right_elbow = df_pose.iloc[row_index]["Right Elbow"]
    right_wrist = df_pose.iloc[row_index]["Right Wrist"]
    left_shoulder = df_pose.iloc[row_index]["Left Shoulder"]
    left_elbow = df_pose.iloc[row_index]["Left Elbow"]
    left_wrist = df_pose.iloc[row_index]["Left Wrist"]
    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    print("Frame: ", x,  "Arm Angle: ", angle)

    #Left side of body is there

    #calculate right arm angle
    # print("Left_Eye: ", df_object.loc[row_indices[0], 'Left Eye'])
    #calculate left arm angle



# print(df[df.Frame==220])
# print(len(df[df.Frame==220]))
# for i in peaks:
#     interest_frames.append(i)
#     # print(ious[i])
# peaks, _ = find_peaks(ious, threshold=0.01)
# print(peaks)