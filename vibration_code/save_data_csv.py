import numpy as np
import csv
import os
from pathlib import Path
from datetime import datetime, timedelta

# Define the directory and the target time range (PST is UTC-8)
target_date = datetime.strptime("2025-02-21", "%Y-%m-%d").date()

# Subject to Change
num_activites = 1

activity_timestamps = [
    "14:11:01",	"14:11:10"
]

activity_timestamps = [datetime.strptime(t, "%H:%M:%S").time() for t in activity_timestamps]

# Define file directory
root_dir = Path("D:\\output\\3331")

# Output directory for CSV files
output_dir = Path("./self_film_data")
output_dir.mkdir(parents=True, exist_ok=True)

# Sampling rate (samples per second)
sampling_rate = 7692.31  # Approximate value
samples_per_second = int(round(sampling_rate))  # Ensure integer samples per second

# Dictionary to store activity data
activity_data = {i: [] for i in range(num_activites)}

# Iterate through the files in the directory
for p in sorted(root_dir.iterdir()):
    try:
        # Extract timestamp from filename (ignoring milliseconds)
        filename_parts = p.stem.split("_")
        date_part = filename_parts[0] 
        time_part = filename_parts[1].split("-")  # ["22", "47", "18", "765393"]
        
        # Construct file timestamp as a datetime object
        file_time_str = f"{time_part[0]}:{time_part[1]}:{time_part[2]}"
        file_datetime_str = f"{date_part} {file_time_str}"
        file_datetime = datetime.strptime(file_datetime_str, "%Y-%m-%d %H:%M:%S")

        # Adjust for UTC-8 to get PST
        file_datetime_pst = file_datetime - timedelta(hours=8)

        # Check if the file timestamp falls within the activity timestamps
        if file_datetime_pst.date() == target_date:
            for i in range(num_activites): 
                if activity_timestamps[i * 2] <= file_datetime_pst.time() <= activity_timestamps[i * 2 + 1]:
                    raw_data = np.load(p)
                    activity_data[i].append(raw_data)
                    break
    
    except (ValueError, IndexError) as e:
        print(f"Skipping file {p.name} due to error: {e}")

# Write each activity's data to a separate CSV file
for i in activity_data:
    output_csv = output_dir / f"couch_s1_{i+1}.csv"
    if activity_data[i]:
        with open(output_csv, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Concatenate data and reshape it into 1-second rows
            activity_array = np.concatenate(activity_data[i])
            num_rows = len(activity_array) // samples_per_second
            reshaped_data = np.array_split(activity_array, num_rows)  # Split into 1-sec chunks
            
            # Write data to CSV
            csv_writer.writerows(reshaped_data)
            print(f"Data successfully written to {output_csv}")
    else:
        print(f"No valid data found for activity {i+1}, skipping file creation.")