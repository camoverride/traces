import numpy as np
import cv2
from pycoral.utils.edgetpu import make_interpreter

# Function to load a memmap .dat file
def load_memmap_file(file_path, shape):
    return np.memmap(file_path, dtype='uint8', mode='r', shape=shape)

# Function to preprocess the frame for the model
def preprocess_frame(frame, target_width, target_height):
    resized_frame = cv2.resize(frame, (target_width, target_height))
    return np.expand_dims(resized_frame.astype(np.float32) / 255.0, axis=0)

# Function to run inference and postprocess the result
def run_inference(interpreter, input_1, input_2):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_1)
    interpreter.set_tensor(input_details[1]['index'], input_2)

    interpreter.invoke()

    output_frame = interpreter.get_tensor(output_details[0]['index'])
    output_frame_uint8 = (output_frame[0] * 255).astype(np.uint8)

    return output_frame_uint8

# Initialize the TFLite model
interpreter = make_interpreter("overlay_model.tflite")
interpreter.allocate_tensors()

# Define the shape of the memmap files (based on your data)
frame_count = 30  # example, adjust based on your data
height = 480  # example, adjust based on your data
width = 640  # example, adjust based on your data
channels = 3  # RGB

memmap_shape = (frame_count, height, width, channels)

# Load two memmap .dat files
file_1_path = "_current_frames.dat"
file_2_path = "_current_frames.dat"

memmap_1 = load_memmap_file(file_1_path, memmap_shape)
memmap_2 = load_memmap_file(file_2_path, memmap_shape)

# Get expected input size for the model
input_details = interpreter.get_input_details()
_, target_height, target_width, _ = input_details[0]['shape']

# Choose a frame index to inspect (e.g., the first frame)
frame_idx = 0
frame_1 = memmap_1[frame_idx]
frame_2 = memmap_2[frame_idx]

# Preprocess the frames
input_1 = preprocess_frame(frame_1, target_width, target_height)
input_2 = preprocess_frame(frame_2, target_width, target_height)

# Run inference
output_frame = run_inference(interpreter, input_1, input_2)

# Display the original frames and the output frame
cv2.imshow("Input Frame 1", frame_1)
cv2.imshow("Input Frame 2", frame_2)
cv2.imshow("Output Frame", output_frame)

# Wait until a key is pressed, then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Clean up
del memmap_1, memmap_2
