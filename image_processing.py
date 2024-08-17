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

def apply_clahe_to_rgb(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE-enhanced L-channel back with the A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to RGB
    final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_frame

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
    
    # Step 2: Apply CLAHE directly on the RGB channels
    processed_frame = apply_clahe_to_rgb(frame)
    
    return processed_frame
