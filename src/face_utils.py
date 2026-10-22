"""Felles kode for opptak, trening og gjenkjenning.

Her ligger personnavnene, 50 %-grensen og funksjonene som finner ansikter,
klargjør bilder, leser personmapper og vurderer modellens forslag.
De andre filene importerer funksjonene slik at de bruker samme regler.
Denne filen trener ikke modellen og åpner ikke kameraet selv.
"""
from pathlib import Path

import cv2
import numpy as np

# Finn prosjektmappen, uansett hvilken mappe programmet startes fra.
ROOT = Path(__file__).resolve().parents[1]

# Modellen bruker tall som personnavn. Olivetti bruker tallene 0 til 39.
PERSON_LABELS = {"Marius": 40, "Endre": 41, "David": 42}
NAMES = {}
for name, label in PERSON_LABELS.items():
    NAMES[label] = name

MIN_CONFIDENCE = 0.50  # Navn vises bare ved OVER 50 %.


def prepare_face(image, box=None):
    """Returner et gråtoneutsnitt på 64 x 64, uten ekstra lysbehandling."""
    if image is None or image.size == 0:
        raise ValueError("Tomt eller uleselig ansiktsbilde.")
    if image.ndim == 3:
        # Et fargebilde har tre fargekanaler. Vi bruker bare gråtoner.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if box is not None:
        x, y, w, h = box
        # Klipp ut rektangelet rundt ansiktet.
        image = image[y:y + h, x:x + w]
    if image.size == 0:
        raise ValueError("Tomt ansiktsutsnitt.")
    return cv2.resize(image, (64, 64))


def face_features(image):
    """Gjør bildet om til de 4096 tallene modellen trenger."""
    face = prepare_face(image)
    pixels = face.astype(np.float32) / 255.0  # Verdier fra 0 til 1.
    return pixels.flatten()  # 64 x 64 blir én rad med 4096 tall.


def detect_faces(detector, gray):
    """Finn ansikter. Dette bestemmer ikke hvem personen er."""
    minimum = max(80, int(gray.shape[0] * 0.25))
    return detector.detectMultiScale(
        cv2.equalizeHist(gray), scaleFactor=1.05, minNeighbors=6,
        minSize=(minimum, minimum),
    )


def decide(model, features):
    """Finn beste forslag, sannsynlighet og om navnet kan vises for hvert bilde."""
    probabilities = model.predict_proba(features)
    labels = []
    confidence = []
    accepted = []

    for scores in probabilities:
        best_index = np.argmax(scores)  # Plassen med høyest sannsynlighet.
        label = model.classes_[best_index]  # Personnummeret på denne plassen.
        score = scores[best_index]

        labels.append(label)
        confidence.append(score)
        accepted.append(score > MIN_CONFIDENCE and label in NAMES)

    return np.array(labels), np.array(confidence), np.array(accepted)


def load_person_images(name):
    """Les PNG-bildene fra én personmappe og gjør dem klare for modellen."""
    person_dir = ROOT / "data" / "faces" / name
    paths = sorted(person_dir.glob("*.png"))
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Kunne ikke lese: {path}")
        images.append(face_features(image))
    return images
