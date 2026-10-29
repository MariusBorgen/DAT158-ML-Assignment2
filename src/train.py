"""Les bilder, tren PCA + SVM, test resultatet og lagre modellen."""
import numpy as np
import joblib

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from face_utils import ROOT, PERSON_LABELS, MIN_CONFIDENCE, select_person_images, decide

def main():
    # ==================== DATA OG TRENING ====================

    # 1. Last inn Olivetti og del i trening og test.
    data = fetch_olivetti_faces()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=30, stratify=data.target,
    )
    X_train = list(X_train)
    X_test = list(X_test)
    y_train = list(y_train)
    y_test = list(y_test)

    # 2. Legg til egne bilder med samme 80/20-fordeling.
    for name, label in PERSON_LABELS.items():
        images = select_person_images(name, require_minimum=False)
        if len(images) < 10:
            print(f"Hopper over {name}: trenger minst 10 unike bilder ({len(images)} funnet).")
            continue
        training, testing = train_test_split(images, test_size=0.2, random_state=30)
        X_train.extend(training)
        X_test.extend(testing)
        y_train.extend([label] * len(training))
        y_test.extend([label] * len(testing))
        print(f"{name}: {len(training)} treningsbilder, {len(testing)} testbilder")

    if not any(label >= 40 for label in y_train):
        raise SystemExit("Ingen registrerte personer har minst 10 unike bilder. Ta bilder før trening.")

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("Tilfeldig bildedeling er en enklere test enn en ny kameraøkt.")

    # 3. Lag PCA- og SVM-modellen.
    model = Pipeline([
        ("pca", PCA(n_components=100, whiten=False, random_state=42)),
        ("svm", SVC(kernel="rbf", C=10, probability=True, random_state=42))
    ])

    # Parameter	Betydning
    # kernel = "rbf"	Gjør det mulig å skille ansikter med ikke-lineære beslutningsgrenser.
    # C = 10	        Kontrollerer hvor kraftig feilklassifisering under trening straffes.



    print("\nTrener modellen...")
    # 4. Tren modellen.
    model.fit(X_train, y_train)

    # ==================== EVALUERING ====================

    # 5. Test med samme regel som kameraet: over 50 %.
    predictions, confidence, accepted = decide(model, X_test)
    correct_counts = []
    wrong_counts = []
    rejected_counts = []
    for name, label in PERSON_LABELS.items():
        correct = 0
        wrong = 0
        rejected = 0
        for i in range(len(y_test)):
            if y_test[i] != label:
                continue  # Dette testbildet tilhører en annen person.
            if not accepted[i]:
                rejected += 1
            elif predictions[i] == label:
                correct += 1
            else:
                wrong += 1
        print(f"{name}: {correct} riktige navn, {wrong} feil navn, {rejected} avvist")
        correct_counts.append(correct)
        wrong_counts.append(wrong)
        rejected_counts.append(rejected)
    print(f"Navn krever over {MIN_CONFIDENCE:.0%}. Tallene under er før avvisning.")

    # Treffsikkerhet for alle personer før avvisning.
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    error_count = 0
    for i in range(len(y_test)):
        if predictions[i] != y_test[i]:
            error_count += 1
    print(f"\nAntall feil: {error_count} av {len(y_test)} testbilder")
    if error_count == 0:
        print("Ingen feilklassifiserte testbilder!")

    # 6. Lagre modellen til gjenkjenningen.
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(
        model,
        model_dir / "face_recognition_custom.joblib"
    )
    print(f"Modellen er lagret i {model_dir / 'face_recognition_custom.joblib'}")

    # 7. Lagre resultatene. plot_results.py leser dem senere.
    train_predictions, _, _ = decide(model, X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    result_dir = ROOT / "reports"
    result_dir.mkdir(exist_ok=True)
    np.savez(
        result_dir / "training_results.npz",
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        correct_counts=correct_counts,
        wrong_counts=wrong_counts,
        rejected_counts=rejected_counts,
        train_accuracy=train_accuracy,
        test_accuracy=accuracy,
        threshold=MIN_CONFIDENCE,
    )
    print("Resultatene er lagret. Kjør python src/plot_results.py for å se grafene.")


if __name__ == "__main__":
    main()
