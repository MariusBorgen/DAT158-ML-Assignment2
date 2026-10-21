
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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


# 1. Last inn Olivetti-datasettet
data = fetch_olivetti_faces()

X_olivetti = data.data
y_olivetti = data.target


# 2. Last inn egne personer
person_labels = {
    "Marius": 40,
    "Endre": 41,
    "David": 42,
}


X_custom = []
y_custom = []

for name, label in person_labels.items():
    face_dir = Path("data/faces") / name
    image_paths = sorted(face_dir.glob("*.png"))

    person_images = []

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Kunne ikke lese: {path}")
            continue

        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float32) / 255.0

        person_images.append(image.flatten())

    if len(person_images) < 10:
        raise ValueError(f"Trenger minst 10 bilder av {name}.")

    X_custom.extend(person_images)
    y_custom.extend([label] * len(person_images))

    print(f"{name}: {len(person_images)} bilder, klasse {label}")

X_custom = np.array(X_custom)
y_custom = np.array(y_custom)

X = np.vstack((X_olivetti, X_custom))
y = np.concatenate((y_olivetti, y_custom))

print(f"Totalt antall bilder: {len(X)}")
print(f"Antall personer: {len(np.unique(y))}")

# 4. Del i trening og testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=30,
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

for name, label in person_labels.items():
    mask = y_test == label

    person_accuracy = accuracy_score(
        y_test[mask],
        predictions[mask],
    )

    print(
        f"{name}: {person_accuracy:.2%} riktig "
        f"av {mask.sum()} testbilder"
    )

# 8. Lagre modellen
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(
    model,
    model_dir / "face_recognition_custom.joblib"
)

figure_dir = Path("reports/figures")
figure_dir.mkdir(parents=True, exist_ok=True)


def person_name(label):
    names = {
        40: "Marius",
        41: "Endre",
        42: "David",
    }
    return names.get(label, f"Person {label}")


def save_and_show(filename):
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


labels = np.unique(y)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

# 9. VIS BILDENE MODELLEN TOK FEIL PÅ
wrong_indices = np.flatnonzero(predictions != y_test)

print(
    f"\nAntall feil: {len(wrong_indices)} "
    f"av {len(y_test)} testbilder"
)

if len(wrong_indices) > 0:
    # Vis opptil 12 feil for at figuren skal være leselig.
    shown_indices = wrong_indices[:12]

    columns = min(4, len(shown_indices))
    rows = int(np.ceil(len(shown_indices) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.5 * rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, i in zip(axes.flat, shown_indices):
        ax.imshow(X_test[i].reshape(64, 64), cmap="gray")
        ax.set_title(
            f"Fasit: {person_name(y_test[i])}\n"
            f"Gjettet: {person_name(predictions[i])}"
        )

    fig.suptitle("Feilklassifiserte testbilder")
    save_and_show("04_feilklassifiserte_bilder.png")

else:
    print("Ingen feilklassifiserte testbilder!")


train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, predictions)

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    ["Treningsbilder", "Testbilder"],
    [train_accuracy * 100, test_accuracy * 100],
    color=["steelblue", "darkorange"],
)

ax.bar_label(bars, fmt="%.1f%%", padding=4)

ax.set(
    title="Treffsikkerhet på trening og test",
    ylabel="Treffsikkerhet (%)",
    ylim=(0, 110),
)

save_and_show("06_trening_mot_test.png")

component_counts = [25, 50, 75, 100, 150]

mean_scores = []
std_scores = []

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42,
)

for count in component_counts:
    candidate = Pipeline([
        ("pca", PCA(
            n_components=count,
            whiten=True,
            random_state=42,
        )),
        ("svm", SVC(kernel="rbf", C=10)),
    ])

    scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        error_score="raise",
    )

    mean_scores.append(scores.mean() * 100)
    std_scores.append(scores.std() * 100)

    print(
        f"{count} PCA-komponenter: "
        f"{scores.mean():.2%} i gjennomsnitt"
    )

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    component_counts,
    mean_scores,
    yerr=std_scores,
    fmt="o-",
    capsize=5,
    color="steelblue",
)

ax.set(
    title="Hvordan påvirker antall PCA-komponenter resultatet?",
    xlabel="Antall PCA-komponenter",
    ylabel="Treffsikkerhet ved kryssvalidering (%)",
    ylim=(0, 105),
)

ax.set_xticks(component_counts)
ax.grid(alpha=0.3)

save_and_show("07_pca_sammenligning.png")

# 11. LAGRE MODELLEN
model_path = model_dir / "face_recognition_custom.joblib"
joblib.dump(model, model_path)



print(f"Figurene er lagret i: {figure_dir}")
print(f"\nModellen er lagret i {model_path}")
