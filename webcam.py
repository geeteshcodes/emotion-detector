import cv2
import tensorflow as tf
import numpy as np
from collections import deque

model = tf.keras.models.load_model("emotion.h5")

class_names = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# PREDICTION SMOOTHING
# -----------------------------
emotion_buffer = deque(maxlen=10)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        if w < 60 or h < 60:
            continue

        # -----------------------------
        # SQUARE FACE CROP
        # -----------------------------
        size = max(w,h)

        cx = x + w//2
        cy = y + h//2

        x1 = max(cx - size//2,0)
        y1 = max(cy - size//2,0)

        face = gray[y1:y1+size, x1:x1+size]

        if face.size == 0:
            continue

        # -----------------------------
        # PREPROCESS
        # -----------------------------
        face = cv2.resize(face,(48,48))
        face = face / 255.0

        face = np.reshape(face,(1,48,48,1))

        # -----------------------------
        # PREDICTION
        # -----------------------------
        preds = model.predict(face,verbose=0)[0]

        emotion_index = np.argmax(preds)
        confidence = preds[emotion_index]

        emotion_buffer.append(emotion_index)

        # majority vote smoothing
        smoothed_index = max(set(emotion_buffer), key=emotion_buffer.count)

        emotion = class_names[smoothed_index]

        # -----------------------------
        # DRAW
        # -----------------------------
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        text = f"{emotion} {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()