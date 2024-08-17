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
    
    # Apply histogram equalization to enhance contrast
    equalized_frame = cv2.equalizeHist(gray_frame)
    
    return equalized_frame


def downscale_image(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height))


def process_image(frame):
    """
    Process an image through a series of adaptive transformations to improve face detection.
    :param frame: The input image frame.
    :return: The processed image frame.
    """
    # Step 1: Downscale the image (if necessary)
    downscaled_frame = downscale_image(frame, scale=0.5)
    
    # Step 2: Check the brightness and adapt
    brightness = np.mean(cv2.cvtColor(downscaled_frame, cv2.COLOR_RGB2HSV)[:, :, 2])
    
    if brightness < 100:  # Image is too dark, increase brightness
        downscaled_frame = increase_brightness(downscaled_frame, value=40)
    elif brightness > 180:  # Image is too bright, apply gamma correction
        downscaled_frame = adjust_gamma(downscaled_frame, gamma=0.6)
    
    # Step 3: Preprocess image (grayscale conversion and histogram equalization)
    processed_frame = preprocess_image(downscaled_frame)
    
    return processed_frame
