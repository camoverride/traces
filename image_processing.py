import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def increase_brightness(frame, value=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    frame_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return frame_bright


def preprocess_image(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_frame = clahe.apply(gray_frame)
    
    # Convert back to RGB format for MediaPipe compatibility
    processed_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2RGB)
    
    return processed_frame


def process_image(frame):
    """
    Process an image through a series of adaptive transformations to improve face detection.
    :param frame: The input image frame.
    :return: The processed image frame.
    """
    # Step 1: Check and adjust brightness if necessary
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 2])
    
    if brightness < 100:  # Image is too dark, increase brightness
        frame = increase_brightness(frame, value=40)
    elif brightness > 180:  # Image is too bright, apply slight gamma correction
        frame = adjust_gamma(frame, gamma=0.8)
    else:
        # Apply a mild gamma correction to standardize appearance
        frame = adjust_gamma(frame, gamma=0.9)
    
    # Step 2: Preprocess image (grayscale conversion, CLAHE, and RGB conversion)
    processed_frame = preprocess_image(frame)
    
    # Convert back to RGB if needed
    if len(processed_frame.shape) == 2 or processed_frame.shape[2] == 1:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
    
    return processed_frame
