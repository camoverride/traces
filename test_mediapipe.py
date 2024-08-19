from picamera2 import Picamera2
import cv2
import mediapipe as mp

FPS = 15

# Initialize the Picamera2
picam2 = Picamera2()
WIDTH, HEIGHT = 640, 480  # for debugging and face detection!
picam2.configure(picam2.create_still_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow("debug", frame)
    if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()
