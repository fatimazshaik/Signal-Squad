import numpy as np
import csv
import os
import itertools
from pathlib import Path
from datetime import datetime, timedelta

# Define the directory and the target time range (PST is UTC-8)
target_date = datetime.strptime("2025-02-12", "%Y-%m-%d").date()

# Sampling rate (samples per second)
sampling_rate = 7692.31  # Approximate value
samples_per_second = int(round(sampling_rate))  # Ensure integer samples per second

# Subject to Change
num_activites = 21
samples_per_window = int(round(0.05 * sampling_rate))  # ≈ 385 samples for 50 ms
threshold = 5060000

# Define Activity Timestamp - Start & End
activity_timestamps = [
    "14:10:16", "14:10:30", "14:10:32", "14:10:45", "14:10:49", "14:11:00", "14:11:01", 
    "14:11:10", "14:11:11", "14:11:22", "14:11:22", "14:11:34", "14:11:36", "14:11:46", 
    "14:11:48", "14:11:58", "14:11:59", "14:12:09", "14:12:10", "14:12:21", "14:12:24", 
    "14:12:35", "14:12:36", "14:12:46", "14:12:46", "14:12:57", "14:12:57", "14:13:08", 
    "14:13:09", "14:13:20", "14:13:21", "14:13:30", "14:13:31", "14:13:43", "14:13:44", 
    "14:13:54", "14:13:56", "14:14:06", "14:14:07", "14:14:16", "14:14:19", "14:14:28"
]

activity_timestamps = [datetime.strptime(t, "%H:%M:%S").time() for t in activity_timestamps]
start_timestamp = None
end_timestamp = None

# Define file directory
root_dir = Path("D:\\output\\3331")

# Output directory for CSV files
output_dir = Path("./self_film_data")
output_dir.mkdir(parents=True, exist_ok=True)

# Dictionary to store activity data (amplitudes)
all_activity_data = {i: [] for i in range(num_activites)}

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
            for mini_activity_count in range(num_activites): 
                mini_activity_decoded = []

                # Go into cooralating file by timestamp
                if activity_timestamps[mini_activity_count * 2] <= file_datetime_pst.time() <= activity_timestamps[mini_activity_count * 2 + 1]:
                    raw_data = np.load(p)
                    
                    # Window the Data
                    num_windows = len(raw_data) // samples_per_window
                    windowed_data_array = np.array_split(raw_data[:num_windows * samples_per_window], num_windows)
                    window_file = output_dir / f"couch_window_amplitudes.csv"
                    with open(window_file, "a", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        for row in windowed_data_array:
                            csv_writer.writerow(row)

                    # Look at each window
                    for window in windowed_data_array:
                        classification = None # 0 or 1
                        labels = []

                        # Majority Vote classification: 
                        for val in window:
                            if val > threshold:
                                labels.append(1)
                            else:
                                labels.append(0)
                        # look at 0s and 1s for each window == determine majority
                        if np.sum(labels) > len(labels) / 2:
                            classification = 1
                        else:
                            classification = 0
                        # append majority classification to larger array --> len(mini_activity_decoded)
                        mini_activity_decoded.append(classification)

                    all_activity_data[mini_activity_count].append(mini_activity_decoded)   

    except (ValueError, IndexError) as e:
        print(f"Skipping file {p.name} due to error: {e}")

all_activity_data_combined = {key: list(itertools.chain(*arrays)) for key, arrays in all_activity_data.items()}
# print(all_activity_data_combined)

# Write each activity's data to a separate CSV file
for activity, data in all_activity_data_combined.items():
    output_csv = output_dir / f"couch_binary_output.csv"
    with open(output_csv, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
              
        # Write data to CSV
        for value in all_activity_data_combined[activity]:
            csv_writer.writerow([value])
            print(f"Data successfully written to {output_csv}")

    # print(f"No valid data found for activity {mini_activity_count+1}, skipping file creation.")