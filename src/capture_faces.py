
import cv2
import time
from pathlib import Path

PERSON_NAME = "Marius"
MAX_IMAGES = 50
SAVE_INTERVAL = 0.3

save_dir = Path("data") / "faces" / PERSON_NAME
save_dir.mkdir(parents=True, exist_ok=True)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Kunne ikke åpne kameraet.")

count = 0
last_capture = 0
capturing = False

print("S = Start | P = Pause | R = Reset | Q = Quit")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) > 0:
            x, y, w, h = max(
                faces, key=lambda face: face[2] * face[3]
            )

            cv2.rectangle(
                frame, (x, y), (x+w, y+h),
                (0, 255, 0), 2
            )

            if capturing and count < MAX_IMAGES:
                if time.time() - last_capture >= SAVE_INTERVAL:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (64, 64))

                    filename = save_dir / f"face_{count:03d}.png"
                    if cv2.imwrite(str(filename), face):
                        count += 1
                        last_capture = time.time()

        if count >= MAX_IMAGES:
            capturing = False

        status = "RECORDING" if capturing else "READY / PAUSED"
        if count >= MAX_IMAGES:
            status = "COMPLETED"

        cv2.putText(
            frame, f"{status} - {count}/{MAX_IMAGES}",
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

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if count < MAX_IMAGES:
                capturing = True
                last_capture = 0

        elif key == ord("p"):
            capturing = False

        elif key == ord("r"):
            capturing = False
            for file in save_dir.glob("face_*.png"):
                file.unlink()
            count = 0
            last_capture = 0

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Saved {count} images in {save_dir}")
