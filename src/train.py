
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Last inn datasettet
data = fetch_olivetti_faces()

X = data.data
y = data.target

# Del data i trening og testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Lag ML-modellen
model = Pipeline([
    ("pca", PCA(n_components=100, whiten=True, random_state=42)),
    ("svm", SVC(kernel="rbf", C=10))
])

# Tren modellen
print("Trener modellen...")
model.fit(X_train, y_train)

# Test modellen
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Lagre modellen
from pathlib import Path

model_path = Path("models")
model_path.mkdir(exist_ok=True)

joblib.dump(model, model_path / "face_recognition_model.joblib")

print("\nModellen er lagret!")


import matplotlib.pyplot as plt
import numpy as np

wrong_indices = np.where(predictions != y_test)[0]

print(f"\nAntall feil: {len(wrong_indices)}")

fig, axes = plt.subplots(
    1, len(wrong_indices),
    figsize=(4 * len(wrong_indices), 4)
)

if len(wrong_indices) > 0:
    axes = np.atleast_1d(axes)

    for ax, i in zip(axes, wrong_indices):
        ax.imshow(X_test[i].reshape(64, 64), cmap="gray")
        ax.set_title(
            f"Actual: {y_test[i]}\nPredicted: {predictions[i]}"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("Ingen feilklassifiserte bilder!")
