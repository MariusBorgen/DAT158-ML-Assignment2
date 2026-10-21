
import cv2
import numpy as np
import joblib

from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Last inn Olivetti-datasettet
data = fetch_olivetti_faces()

X_olivetti = data.data
y_olivetti = data.target

# 2. Last inn bildene dine
face_dir = Path("data/faces/Marius")
image_paths = sorted(face_dir.glob("*.png"))

X_custom = []

for path in image_paths:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    image = cv2.resize(image, (64, 64))

    # Samme pixelverdiområde som Olivetti: 0-1
    image = image.astype(np.float32) / 255.0

    X_custom.append(image.flatten())

if len(X_custom) < 10:
    raise ValueError("Trenger minst 10 bilder av Marius.")

X_custom = np.array(X_custom)

# Marius får klasse 40
y_custom = np.full(len(X_custom), 40)

# 3. Kombiner datasettene
X = np.vstack((X_olivetti, X_custom))
y = np.concatenate((y_olivetti, y_custom))

print(f"Totalt antall bilder: {len(X)}")
print(f"Antall personer: {len(np.unique(y))}")

# 4. Del i trening og testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Lag ML-modellen
model = Pipeline([
    ("pca", PCA(n_components=100, whiten=True, random_state=42)),
    ("svm", SVC(kernel="rbf", C=10))
])

# 6. Tren modellen
print("\nTrener modellen...")
model.fit(X_train, y_train)

# 7. Test modellen
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Vis resultatene for Marius
mask = y_test == 40

marius_accuracy = accuracy_score(
    y_test[mask],
    predictions[mask]
)

print(f"\nMarius accuracy: {marius_accuracy * 100:.2f}%")

# 8. Lagre modellen
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(
    model,
    model_dir / "face_recognition_custom.joblib"
)

print("\nModellen er lagret!")
