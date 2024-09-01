# Imports
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import cv2
from fer import FER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loading the pre-trained emotion detection model
detector = FER()

# Function to detect emotion in a given frame
def detect_emotion(frame):

    # Detecting emotions in the frame
    emotions = detector.detect_emotions(frame)
    
    # Displaying detected emotions
    for face in emotions:
        (x, y, w, h) = face['box']
        emotion_label = max(face['emotions'], key=face['emotions'].get)
        cv2.putText(frame, f"{emotion_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
    
    return frame

# Capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Reading a frame from the webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Flipping the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detecting emotion in the frame
    frame = detect_emotion(frame)
    
    # Displaying the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Breaking the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the video capturing object and closing all windows
cap.release()
cv2.destroyAllWindows()
