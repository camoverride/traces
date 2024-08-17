import numpy as np
import os
import cv2



# Function to infer the number of frames in the memmap file
def infer_frame_count(file_path, height, width, channels):
    # Calculate the size of a single frame in bytes
    frame_size = height * width * channels
    
    # Get the total size of the file in bytes
    file_size = os.path.getsize(file_path)
    
    # Infer the number of frames
    frame_count = file_size // frame_size
    
    return frame_count

# Function to load a specific frame from a memmap .dat file
def load_memmap_frame(file_path, height, width, channels, frame_idx):
    # Infer the number of frames
    frame_count = infer_frame_count(file_path, height, width, channels)
    
    # Define the shape of the memmap
    memmap_shape = (frame_count, height, width, channels)
    
    # Load the memmap file
    memmap_file = np.memmap(file_path, dtype='uint8', mode='r', shape=memmap_shape)
    frame = memmap_file[frame_idx]
    del memmap_file  # Clean up memory
    return frame

# Example usage:
# Path to the memmap .dat file
file_path = "_current_frames.dat"

# Known dimensions (height, width, channels)
height = 1920  # Update this based on your data
width =  1080 # Update this based on your data
channels = 3  # Update this based on your data (3 for RGB, 1 for grayscale, etc.)

# Index of the frame you want to display
frame_idx = 10  # Adjust this to choose a different frame

# Load and display the frame
frame = load_memmap_frame(file_path, height, width, channels, frame_idx)

# Display the frame using OpenCV
cv2.imshow("Frame", frame)

# Wait for 5000 ms (5 seconds) before closing the window
cv2.waitKey(5000)
cv2.destroyAllWindows()
