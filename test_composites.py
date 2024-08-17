import numpy as np
import cv2
from pycoral.utils.edgetpu import make_interpreter
import os

# Function to infer the number of frames based on file size and dimensions
def infer_frame_count(file_path, height, width, channels):
    frame_size = height * width * channels
    file_size = os.path.getsize(file_path)
    frame_count = file_size // frame_size
    return frame_count

# Function to load a memmap .dat file with inferred shape
def load_memmap_file(file_path, height, width, channels):
    frame_count = infer_frame_count(file_path, height, width, channels)
    shape = (frame_count, height, width, channels)
    return np.memmap(file_path, dtype='uint8', mode='r', shape=shape)

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    return np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)

# Function to run inference and postprocess the result
def run_inference(interpreter, input_1, input_2):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_1)
    interpreter.set_tensor(input_details[1]['index'], input_2)

    interpreter.invoke()

    output_frame = interpreter.get_tensor(output_details[0]['index'])

    # Debug: Print the output frame's shape and min/max values
    print(f"Output Frame Shape: {output_frame.shape}")
    print(f"Output Frame Data Type: {output_frame.dtype}")
    print(f"Output Frame Min Value: {output_frame.min()}")
    print(f"Output Frame Max Value: {output_frame.max()}")

    output_frame_uint8 = (output_frame[0] * 255).astype(np.uint8)

    return output_frame_uint8

# Initialize the TFLite model
interpreter = make_interpreter("alpha_blending_model.tflite")  # Update with your model's file name
interpreter.allocate_tensors()

# Known dimensions (height, width, channels)
height = 1920  # Update this based on your data
width = 1080   # Update this based on your data
channels = 3   # RGB

# Load two memmap .dat files with inferred frame count
file_1_path = "_current_frames.dat"
file_2_path = "current_frames.dat"

memmap_1 = load_memmap_file(file_1_path, height, width, channels)
memmap_2 = load_memmap_file(file_2_path, height, width, channels)

# Get expected input size for the model (in case input resizing is needed)
input_details = interpreter.get_input_details()
_, target_height, target_width, _ = input_details[0]['shape']

# If the input size to the model does not require resizing, you can use:
# target_height = height
# target_width = width

# Choose a frame index to inspect (ensure it is within range)
frame_idx = min(50, len(memmap_1) - 1)  # Ensure frame_idx is within the valid range
frame_1 = memmap_1[frame_idx]
frame_2 = memmap_2[frame_idx]

# Preprocess the frames
input_1 = preprocess_frame(frame_1)
input_2 = preprocess_frame(frame_2)

# Run inference
output_frame = run_inference(interpreter, input_1, input_2)

# Display the original frames and the output frame
cv2.imshow("Input Frame 1", frame_1)
cv2.waitKey(1000)
cv2.destroyAllWindows()

cv2.imshow("Input Frame 2", frame_2)
cv2.waitKey(1000)
cv2.destroyAllWindows()

cv2.imshow("Output Frame", output_frame)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# Clean up
del memmap_1, memmap_2
