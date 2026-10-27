"""Kontroller at analysen holder valideringsdata utenfor PCA-treningen."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare_models
from compare_models import load_training_data
from face_utils import PERSON_LABELS, load_person_images


class AnalysisChecks(unittest.TestCase):
    def test_pca_is_fitted_only_on_each_training_fold(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 16)).astype(np.float32)
        y = np.repeat(np.arange(3), 20)
        splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
        real_pca = compare_models.PCA
        pca_models = []

        def track_pca(**parameters):
            pca = real_pca(**parameters)
            pca.fit_transform = unittest.mock.Mock(wraps=pca.fit_transform)
            pca.transform = unittest.mock.Mock(wraps=pca.transform)
            pca_models.append(pca)
            return pca

        with patch.object(compare_models, "COMPONENTS", [2, 4]):
            with patch.object(compare_models, "PCA", side_effect=track_pca):
                with threadpool_limits(limits=1):
                    results = compare_models.run_experiments(X, y, splits)

        self.assertEqual(len(pca_models), 3 * 2 * 2)
        self.assertEqual(len(results), 3 * 2 * 2 * 4)
        for fold, (training, validation) in enumerate(splits):
            self.assertEqual(len(np.intersect1d(training, validation)), 0)
            for pca in pca_models[fold * 4:(fold + 1) * 4]:
                pca.fit_transform.assert_called_once()
                pca.transform.assert_called_once()
                np.testing.assert_array_equal(pca.fit_transform.call_args.args[0], X[training])
                np.testing.assert_array_equal(pca.transform.call_args.args[0], X[validation])

    def test_training_data_matches_existing_train_script(self):
        data = fetch_olivetti_faces(download_if_missing=False)
        X, _, y, _ = train_test_split(data.data, data.target, test_size=.2,
                                      random_state=30, stratify=data.target)
        X, y = list(X), list(y)
        for name, label in PERSON_LABELS.items():
            images, seen = [], set()
            for image in load_person_images(name):
                if image.tobytes() not in seen:
                    images.append(image)
                    seen.add(image.tobytes())
            training, _ = train_test_split(images, test_size=.2, random_state=30)
            X.extend(training)
            y.extend([label] * len(training))
        actual_X, actual_y, _ = load_training_data()
        np.testing.assert_array_equal(actual_X, np.asarray(X))
        np.testing.assert_array_equal(actual_y, np.asarray(y))


if __name__ == "__main__":
    unittest.main()
