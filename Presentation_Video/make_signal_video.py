import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load background video
video_path = "/home/jsguo/EEC174/Signal-Squad/Presentation/couch7.mov"  # Replace with your video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Generate synthetic vibration data (Replace with real data)
fs = 100  # Sampling frequency in Hz
duration = total_frames // fps  # Match video duration
t = np.linspace(0, duration, fs * duration)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t))

# Setup video writer with high resolution
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Create figure for signal plot
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)  # Higher resolution
ax.set_xlim(0, duration)
ax.set_ylim(-2, 2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Vibration Signal Over Time')
line, = ax.plot([], [], 'b', linewidth=2)

# Process frames
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(t):
        break

    # Update signal plot
    line.set_data(t[:frame_idx], signal[:frame_idx])
    plt.savefig("signal_frame.png", bbox_inches='tight', dpi=300)  # High-res image
    
    # Load the signal plot as an image
    signal_img = cv2.imread("signal_frame.png")
    signal_img = cv2.resize(signal_img, (frame_width, frame_height // 4))  # Resize to fit
    signal_img = cv2.flip(signal_img, 0)  # Flip to correct upside-down issue

    # Combine with video frame
    combined_frame = frame.copy()
    combined_frame[-signal_img.shape[0]:, :] = signal_img  # Overlay at bottom

    out.write(combined_frame)
    frame_idx += 1

# Cleanup
cap.release()
out.release()
os.remove("signal_frame.png")
print(f"Output saved as {output_path}")
