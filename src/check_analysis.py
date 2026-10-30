"""Kontroller at analysen holder valideringsdata utenfor PCA-treningen."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) # Legg til src-mappen i sys.path for å importere compare_models
import compare_models
from compare_models import load_training_data # Importer load_training_data fra compare_models
from face_utils import PERSON_LABELS, load_person_images # Importer load_person_images fra face_utils

# Klasse for å teste at PCA kun trenes på de riktige treningsfoldene og at treningsdataene samsvarer med eksisterende treningsskript.
class AnalysisChecks(unittest.TestCase):

    # Test at PCA kun trenes på treningsfoldene og ikke på valideringsfoldene.
    def test_pca_is_fitted_only_on_each_training_fold(self):
        rng = np.random.default_rng(42) #Seed for samme tilfeldige data hver gang
        X = rng.normal(size=(60, 16)).astype(np.float32) # Lag 60 tilfeldige bilder med 16 features, og gjør dem til float32 
        y = np.repeat(np.arange(3), 20) # Lag 3 klasser med 20 bilder hver
        splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y)) # Lag 3 fold med stratified k-fold. To folds til trening og en til validering
        real_pca = compare_models.PCA # Lagre den originale PCA-klassen for senere bruk
        pca_models = [] # Liste for å lagre PCA-modeller som opprettes under testen

        # Sporer hver PCA-instans som opprettes, og erstatter fit_transform og transform med mock-objekter for å spore kallene.
        def track_pca(**parameters):
            pca = real_pca(**parameters)
            pca.fit_transform = unittest.mock.Mock(wraps=pca.fit_transform)
            pca.transform = unittest.mock.Mock(wraps=pca.transform)
            pca_models.append(pca)
            return pca

        # midlertidig endre instillingene for compare_models for å bruke 2 og 4 komponenter, og patch PCA-klassen for å spore kallene.
        with patch.object(compare_models, "COMPONENTS", [2, 4]):
            with patch.object(compare_models, "PCA", side_effect=track_pca):
                with threadpool_limits(limits=1):
                    results = compare_models.run_experiments(X, y, splits)

        # Sjekk at vi har opprettet PCA-modeller for hver fold og hver komponent, og at resultatene samsvarer med forventningene.
        self.assertEqual(len(pca_models), 3 * 2 * 2)
        self.assertEqual(len(results), 3 * 2 * 2 * 4)

        # Sjekk at trenings- og valideringsdataene ikke overlapper.
        for fold, (training, validation) in enumerate(splits):
            self.assertEqual(len(np.intersect1d(training, validation)), 0)

            # Sjekk at fit_transform kun kalles på treningsdataene og transform kun kalles på valideringsdataene for hver PCA-modell.
            for pca in pca_models[fold * 4:(fold + 1) * 4]:
                pca.fit_transform.assert_called_once()
                pca.transform.assert_called_once()
                np.testing.assert_array_equal(pca.fit_transform.call_args.args[0], X[training])
                np.testing.assert_array_equal(pca.transform.call_args.args[0], X[validation])

    # Test at treningsdataene som brukes i analysen samsvarer med de som brukes i det eksisterende treningsskriptet.
    def test_training_data_matches_existing_train_script(self):

        # forsøk å hente Olivetti-datasettet uten å laste det ned hvis det ikke finnes lokalt. Hvis det ikke finnes, hopp over testen.
        try:
            data = fetch_olivetti_faces(download_if_missing=False)
        except OSError:
            self.skipTest("Olivetti er ikke lastet ned lokalt.")

        # Sjekk at vi har minst 10 unike treningsbilder for hver person i PERSON_LABELS. Hvis ikke, hopp over testen.
        if any(len(load_person_images(name)) < 10 for name in PERSON_LABELS):
            self.skipTest("Private treningsbilder mangler.")
        
        # Del datasettet i trening og test.
        # 20% skal være testdata.
        X, _, y, _ = train_test_split(data.data, data.target, test_size=.2,
                                      random_state=30, stratify=data.target)
        X, y = list(X), list(y)

        # Legg til private treningsbilder for hver person i PERSON_LABELS.
        for name, label in PERSON_LABELS.items():
            images, seen = [], set()
            for image in load_person_images(name):

                # Ta kun med unike bilder.
                if image.tobytes() not in seen:
                    images.append(image)
                    seen.add(image.tobytes())
            
            training, _ = train_test_split(images[:100], test_size=.2, random_state=30) #deler opp i 80% trening og 20% test.
            X.extend(training) # Legg til treningsbildene i X
            y.extend([label] * len(training)) # Legg til tilsvarende labels i y
        actual_X, actual_y, _ = load_training_data() # Hent treningsdataene som brukes i analysen
        np.testing.assert_array_equal(actual_X, np.asarray(X)) # Sjekk at treningsdataene som brukes i analysen samsvarer med de som brukes i det eksisterende treningsskriptet.
        np.testing.assert_array_equal(actual_y, np.asarray(y)) # Sjekk at treningslabelene som brukes i analysen samsvarer med de som brukes i det eksisterende treningsskriptet.


if __name__ == "__main__":
    unittest.main()
