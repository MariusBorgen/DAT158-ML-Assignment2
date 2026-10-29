"""Felles kode for opptak, trening og gjenkjenning.

Her ligger personnavnene, 50 %-grensen og funksjonene som finner ansikter,
klargjør bilder, leser personmapper og vurderer modellens forslag.
De andre filene importerer funksjonene slik at de bruker samme regler.
Denne filen trener ikke modellen og åpner ikke kameraet selv.
"""
from pathlib import Path
import json
import re
import threading

import cv2
import numpy as np

# Finn prosjektmappen, uansett hvilken mappe programmet startes fra.
ROOT = Path(__file__).resolve().parents[1]

# Olivetti bruker ID-ene 0–39. Lokale personer starter på 40.
PERSON_LABELS = {}
NAMES = {}
_registry_lock = threading.Lock()


def refresh_people():
    path = ROOT / "data" / "people.json"
    people = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if (not isinstance(people, dict)
            or any(not isinstance(name, str) or not valid_person_name(name)
                   or type(label) is not int or label < 40
                   for name, label in people.items())
            or len(set(people.values())) != len(people)):
        raise ValueError("Ugyldig personregister i data/people.json.")
    PERSON_LABELS.clear()
    PERSON_LABELS.update(people)
    NAMES.clear()
    NAMES.update({label: name for name, label in people.items()})
    return PERSON_LABELS


def valid_person_name(name):
    return (bool(name) and name == name.strip() and len(name) <= 100
            and not re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            and not name.endswith(".")
            and name.split(".")[0].upper() not in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))})


def register_person(name):
    name = name.strip()
    if not valid_person_name(name):
        raise ValueError("Skriv et gyldig navn uten filsti eller spesialtegn.")
    with _registry_lock:
        refresh_people()
        if any(existing.casefold() == name.casefold() for existing in PERSON_LABELS):
            raise ValueError("Personen er allerede registrert.")
        people = dict(PERSON_LABELS)
        people[name] = max(people.values(), default=39) + 1
        path = ROOT / "data" / "people.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(people, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        refresh_people()
    return name


refresh_people()

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


def select_person_images(name, require_minimum=True):
    """Første 100 unike bilder i filnavnrekkefølge; felles for trening og analyse."""
    images, seen = [], set()
    for image in load_person_images(name):
        key = image.tobytes()
        if key not in seen:
            images.append(image)
            seen.add(key)
    if require_minimum and len(images) < 10:
        raise ValueError(f"Trenger minst 10 unike bilder av {name}.")
    return images[:100]


def load_model(path=None):
    """Les bare en betrodd modell mottatt privat eller trent lokalt."""
    import joblib
    path = Path(path) if path is not None else ROOT / "models/face_recognition_custom.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"Modellfilen mangler: {path}. Motta modellen privat eller kjør python src/train.py."
        )
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Modellen mangler predict_proba. Tren modellen på nytt.")
    if not {int(label) for label in model.classes_ if label >= 40}.issubset(NAMES):
        raise ValueError("Modellens personnummer mangler i personregisteret. Tren modellen på nytt.")
    return model
