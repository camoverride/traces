from picamera2 import Picamera2
import cv2
import mediapipe as mp



FPS = 15

# Initialize the Picamera2
picam2 = Picamera2()
WIDTH, HEIGHT = 640, 480 # for debugging and face detectioN!
picam2.configure(picam2.create_still_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


with mp_face_detection.FaceDetection as face_detection:
    while True:
        frame = picam2.capture_array()
        results = face_detection.process(frame)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        cv2.imshow("debug", frame)
        cv2.waitKey(int(1000 / FPS))