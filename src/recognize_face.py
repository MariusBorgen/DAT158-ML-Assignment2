
import cv2
import joblib
import numpy as np

# Last inn den trente modellen
model = joblib.load("models/face_recognition_custom.joblib")

# Ansiktsdeteksjon
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Kunne ikke åpne kameraet")

print("Trykk SPACE for å gjenkjenne ansiktet")
print("Trykk Q for å avslutte")

prediction_text = "Press SPACE to recognize"

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) > 0:
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            prediction_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Welcome Home - Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if len(faces) == 0:
                prediction_text = "No face detected"
                continue

            # Bruk største ansikt
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))

            # Normaliser og gjør klar for modellen
            face = face.astype(np.float32) / 255.0
            features = face.flatten().reshape(1, -1)

            prediction = int(model.predict(features)[0])

            if prediction == 40:
                prediction_text = "Welcome home, Marius!"
            else:
                prediction_text = f"Predicted: Olivetti #{prediction}"

            print(prediction_text)

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
