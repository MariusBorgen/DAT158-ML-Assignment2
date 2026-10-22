import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import face_utils as faces


class FaceTests(unittest.TestCase):
    def test_probability_boundary_and_class_mapping(self):
        class Model:
            classes_ = np.array([42, 0, 40])

            def predict_proba(self, features):
                return np.array([
                    [.5, .2, .3], [.51, .2, .29], [.1, .8, .1],
                ])

        labels, _, accepted = faces.decide(Model(), np.zeros((3, 4096)))
        np.testing.assert_array_equal(labels, [42, 42, 0])
        np.testing.assert_array_equal(accepted, [False, True, False])

    def test_saved_crop_matches_camera_features(self):
        image = np.arange(120 * 120, dtype=np.uint8).reshape(120, 120)
        crop = faces.prepare_face(image, (10, 20, 70, 80))
        _, encoded = cv2.imencode(".png", crop)
        saved = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(faces.face_features(crop), faces.face_features(saved))
        self.assertEqual(faces.face_features(saved).shape, (4096,))

    def test_loads_person_folder_only(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(faces, "ROOT", Path(tmp)):
            person = Path(tmp) / "data/faces/Marius"
            for relative, value in [("a.png", 20), ("b.png", 40),
                                    ("ignored/b.png", 200)]:
                path = person / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                self.assertTrue(cv2.imwrite(str(path), np.full((64, 64), value, np.uint8)))
            training = faces.load_person_images("Marius")
            self.assertEqual(len(training), 2)
            self.assertTrue(all(image.mean() < .2 for image in training))
            self.assertEqual(faces.load_person_images("Endre"), [])


if __name__ == "__main__":
    unittest.main()
