"""Finn ett ansikt i kameraet og vis navn hvis modellen godtar treffet."""
import time

import cv2
import joblib
from face_utils import ROOT, NAMES, MIN_CONFIDENCE, prepare_face, face_features, detect_faces, decide
# Les modellen som train.py har lagret.
model = joblib.load(ROOT / "models/face_recognition_custom.joblib")

if not hasattr(model, "predict_proba"):
    raise RuntimeError(
        "Modellen mangler sannsynlighetsestimat. "
        "Kjør python src/train.py og start gjenkjenningen på nytt."
    )

SCAN_INTERVAL = 0.3  # Sekunder mellom hver prediksjon.

# Detektoren finner hvor ansiktet er. Modellen bestemmer hvem det er.
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Kunne ikke åpne kameraet.")

recognized = False
last_scan = 0
message = "Looking for face..."

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        # Ett kamerabilde om gangen, gjort om til gråtoner.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        if len(faces) != 1:
            recognized = False

            if len(faces) == 0:
                message = "Looking for face..."
            else:
                message = "Only one person at a time"

        else:
            x, y, w, h = faces[0]

            if time.monotonic() - last_scan >= SCAN_INTERVAL:
                last_scan = time.monotonic()

                # Samme bildeformat som under trening.
                face = prepare_face(gray, (x, y, w, h))
                # Modellen forventer en liste med bilder, selv når vi har bare ett.
                features = [face_features(face)]

                # Bare forhåndsvisningen forstørres. Modellen får 64 x 64.
                preview = cv2.resize(
                    face,
                    (320, 320),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow("Bildet modellen ser", preview)

                # Bruk klassene i modellen: sannsynlighetskolonnene er ikke navn.
                labels, confidences, accepted = decide(model, features)
                # [0] henter resultatet for det ene bildet vi sendte inn.
                prediction = int(labels[0])
                confidence = float(confidences[0])

                recognized = bool(accepted[0])
                if confidence <= MIN_CONFIDENCE:
                    message = f"Uncertain ({confidence:.1%}) - try again"

                elif prediction not in NAMES:
                    message = "Person not recognized"

                else:
                    name = NAMES[prediction]
                    message = f"Welcome home, {name}! ({confidence:.1%})"

            color = (0, 200, 255)  # Gul mens treffet ikke er godkjent.
            if recognized:
                color = (0, 255, 0)  # Grønn når navnet vises.

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
    # Lukk kamera og vinduer, også hvis en feil oppstår.
    camera.release()
    cv2.destroyAllWindows()
