import csv

filename_iou = "path_vector/iou_output.csv"
filename_path = "path_vector/path_output_all4.csv"

data_iou = []
data_path = []

with open(filename_iou, mode="r") as file:
    reader = csv.reader(file)
    
    # Skip the header row
    next(reader)  
    
    # Store the remaining rows in a list
    for row in reader:
        data_iou.append(row)

with open(filename_path, mode="r") as file:
    reader = csv.reader(file)
    
    # Skip the header row
    next(reader)  
    
    # Store the remaining rows in a list
    for row in reader:
        data_path.append(row)

print(data_iou)
print(data_path)

action_map = {
    'Sitting on Couch': 'couch action',
    'Sitting on Table': 'chair action',
    'Opening Fridge': 'fridge action',
    'Laying on Bed': 'bed action'
}

merged_result = []

for i in range (len(data_iou)):
    start_iou = int(data_iou[i][1])
    end_iou = int(data_iou[i][2])

    start_path = int(data_path[i][1])
    end_path = int(data_path[i][2])
    if (action_map[data_iou[i][0]] == data_path[i][0]):
        print("match")
        action_type = data_path[i][0]
        if(action_type == 'chair action'):
            merged_result.append([action_type, start_iou, end_iou])
        elif(action_type == 'fridge action'):
            merged_result.append([action_type, start_path, end_path])
        else:
            merged_result.append([action_type, (start_path + start_iou) // 2, (end_path + end_iou) // 2])

    print(start_iou, end_iou, start_path, end_path)
# Print results
print(merged_result)


# Save result
filename = "path_vector/merged_result.csv"
headers = ["Action Type", "Start Time", "End Time"]

with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers first
    writer.writerows(merged_result)  # Write all rows at once

print(f"CSV file '{filename}' has been created with {len(merged_result)} rows.")