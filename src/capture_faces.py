"""Ta ansiktsbilder og lagre dem i mappen til valgt person."""
import cv2
import time
from datetime import datetime
import argparse
from face_utils import ROOT, PERSON_LABELS, prepare_face, detect_faces

# Innstillinger du kan endre.
PERSON_NAME = "Marius"
MAX_IMAGES = 150
SAVE_INTERVAL = 0.3

# --person lar deg velge navn fra terminalen uten å endre denne filen.
parser = argparse.ArgumentParser()
parser.add_argument("--person", choices=PERSON_LABELS, default=PERSON_NAME)
args = parser.parse_args()
save_dir = ROOT / "data" / "faces" / args.person
save_dir.mkdir(parents=True, exist_ok=True)
session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
session_files = []  # Bare filene som lagres i denne kjøringen.

# OpenCV har en ferdig detektor som finner ansiktsrektangelet.
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)  # 0 er standardkameraet.

if not camera.isOpened():
    raise RuntimeError("Kunne ikke åpne kameraet.")

last_capture = 0
capturing = False

print("S = Start | P = Pause | R = Reset | Q = Quit")
print(f"Saving to {save_dir}. R resets only this session.")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # frame er ett kamerabilde. gray er det samme bildet i gråtoner.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        if len(faces) == 1:
            x, y, w, h = faces[0]  # Venstre kant, topp, bredde og høyde.

            cv2.rectangle(
                frame, (x, y), (x+w, y+h),
                (0, 255, 0), 2
            )

            enough_time = time.monotonic() - last_capture >= SAVE_INTERVAL
            if capturing and len(session_files) < MAX_IMAGES and enough_time:
                face = prepare_face(gray, (x, y, w, h))
                # Tidsstempel beskytter gamle opptak mot overskriving.
                filename = save_dir / f"{session_id}_face_{len(session_files):03d}.png"
                if cv2.imwrite(str(filename), face):
                    session_files.append(filename)
                    last_capture = time.monotonic()

        if len(session_files) >= MAX_IMAGES:
            capturing = False

        status = "READY / PAUSED"
        if capturing:
            status = "RECORDING"
        if len(session_files) >= MAX_IMAGES:
            status = "COMPLETED"
        elif len(faces) != 1:
            status = "Need exactly one face"

        cv2.putText(
            frame, f"{status} - {len(session_files)}/{MAX_IMAGES}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            frame, "S: Start  P: Pause  R: Reset  Q: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )

        cv2.imshow("Capture Faces - DAT158", frame)

        key = cv2.waitKey(1) & 0xFF  # Les en tast uten å stoppe kameravisningen.

        if key == ord("s"):
            if len(session_files) < MAX_IMAGES:
                capturing = True
                last_capture = 0

        elif key == ord("p"):
            capturing = False

        elif key == ord("r"):
            capturing = False
            # R fjerner bare denne øktens bilder, aldri eldre bilder.
            for file in session_files:
                file.unlink()
            session_files.clear()
            last_capture = 0

        elif key == ord("q"):
            break

finally:
    # Frigjør kameraet også hvis programmet stopper med en feil.
    camera.release()
    cv2.destroyAllWindows()

print(f"Saved {len(session_files)} images in {save_dir}")
