import time

import cv2
import joblib
import numpy as np


model = joblib.load("models/face_recognition_custom.joblib")

names = {
    40: "Marius",
    41: "Endre",
    42: "David",
}

REQUIRED_MATCHES = 10
SCAN_INTERVAL = 0.3

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Kunne ikke åpne kameraet.")

last_prediction = None
match_count = 0
last_scan = 0
message = "Looking for face..."

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection_image = cv2.equalizeHist(gray)

        min_face_size = max(80, int(gray.shape[0] * 0.25))

        faces = detector.detectMultiScale(
            detection_image,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(min_face_size, min_face_size),
        )

        # Krev nøyaktig ett ansikt i bildet.
        if len(faces) != 1:
            last_prediction = None
            match_count = 0

            message = (
                "Looking for face..."
                if len(faces) == 0
                else "Only one person at a time"
            )

        else:
            x, y, w, h = faces[0]

            if time.monotonic() - last_scan >= SCAN_INTERVAL:
                last_scan = time.monotonic()

                # Samme bildeformat som under trening.
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (64, 64))
                features = face.astype(np.float32) / 255.0
                features = features.flatten().reshape(1, -1)

                # Forstørr bare visningen. Modellen får fortsatt 64 × 64.
                preview = cv2.resize(
                    face,
                    (320, 320),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow("Bildet modellen ser", preview)

                prediction = int(model.predict(features)[0])

                if prediction not in names:
                    last_prediction = None
                    match_count = 0
                    message = "Person not recognized"

                else:
                    if prediction == last_prediction:
                        match_count = min(
                            match_count + 1,
                            REQUIRED_MATCHES,
                        )
                    else:
                        last_prediction = prediction
                        match_count = 1

                    name = names[prediction]

                    if match_count >= REQUIRED_MATCHES:
                        message = f"Welcome home, {name}!"
                    else:
                        message = (
                            f"Checking {name}: "
                            f"{match_count}/{REQUIRED_MATCHES}"
                        )

            color = (
                (0, 255, 0)
                if match_count >= REQUIRED_MATCHES
                else (0, 200, 255)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2,
            )

        cv2.putText(
            frame,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Welcome Home - Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()