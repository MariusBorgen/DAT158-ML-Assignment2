"""Sammenlign PCA og fire algoritmer uten å bruke testsettet.

Les main() for oversikten og run_experiments() for selve forsøket.
Grafene og lagringen er samlet i egne funksjoner lenger ned.
"""
import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # Lagre grafer uten å åpne mange vinduer.
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from face_utils import ROOT, PERSON_LABELS, select_person_images

# ==================== INNSTILLINGER ====================

COMPONENTS = [25, 50, 75, 100, 125, 150]
WHITENING = [False, True]
SEED = 42


# ==================== OVERSIKT: HER STARTER ANALYSEN ====================

def main():
    # Valgfrie argumenter. Vanlig kjøring: python src/compare_models.py
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--illustration-labels", type=int, nargs=2, default=[0, 1])
    args = parser.parse_args()

    # 1. Hent bare de 80 % som train.py bruker til trening.
    X_train, y_train, counts = load_training_data()
    if not 2 <= args.folds <= min(counts.values()):
        parser.error("Antall folder må være mellom 2 og minste antall treningsbilder per person.")
    if (len(set(args.illustration_labels)) != 2
            or not set(args.illustration_labels).issubset(set(y_train))):
        parser.error("Illustrasjonen krever to forskjellige personnummer som finnes i dataene.")

    # 2. Lag samme stratifiserte folder for alle forsøkene.
    # Stratifisert betyr at hver fold beholder omtrent samme klassefordeling.
    cross_validation = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=SEED,
    )
    splits = list(cross_validation.split(X_train, y_train))
    smallest_fold = min(len(training) for training, _ in splits)
    if smallest_fold <= max(COMPONENTS):
        parser.error("For få treningsbilder i foldene til 150 PCA-komponenter.")

    # 3. Tren og valider alle modellene. Én beregningstråd gir stabil kjøretid.
    with threadpool_limits(limits=1):
        results = run_experiments(X_train, y_train, splits)
        summary = summarize_results(results)

        # 4. Lagre nye resultater først når forsøkene er ferdige.
        output, figures = prepare_folders()
        write_csv(output / "fold_results.csv", results)
        write_csv(output / "summary.csv", summary)
        save_metadata(X_train, y_train, counts, splits, args.illustration_labels, output)
        report(summary, output)

        # 5. Tegn figurene. Ingen av disse figurene bruker testsettet.
        plt.rcParams.update({"font.size": 11})
        plot_pca_comparison(summary, figures)
        plot_algorithm_comparison(summary, figures)
        plot_eigenfaces(X_train, figures)
        plot_svm_illustration(X_train, y_train, figures, args.illustration_labels)

    print(f"\nÅpne resultatet: {output / 'analysis.md'}")
    print(f"Grafene ligger i: {figures}")


# ==================== 1. HENT TRENINGSDATA ====================

def load_training_data():
    """Bruk samme bilder og 80/20-deling som train.py. Hold testdelen utenfor."""
    data = fetch_olivetti_faces(download_if_missing=False)
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target,
        test_size=0.2, random_state=30, stratify=data.target,
    )
    X_train = list(X_train)
    y_train = list(y_train)

    for name, label in PERSON_LABELS.items():
        images = select_person_images(name, require_minimum=False)
        if len(images) < 10:
            print(f"Hopper over {name}: trenger minst 10 unike bilder ({len(images)} funnet).")
            continue
        training, _ = train_test_split(images, test_size=0.2, random_state=30)
        X_train.extend(training)
        y_train.extend([label] * len(training))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    counts = {}
    for label in np.unique(y_train):
        counts[str(label)] = int(np.sum(y_train == label))

    return X_train, y_train, counts


# ==================== 2. VELG ALGORITMER ====================

def classifiers():
    """Lag nye, utrente klassifikatorer til hvert forsøk."""
    # C=10 beholdes for begge SVM-modellene. C og gamma kan tunes senere.
    # Vi måler vanlige klasseforslag, uten kameraets 50 %-avvisning.
    return {
        "SVM RBF": SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED),
        "SVM Linear": SVC(kernel="linear", C=10, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(
            C=1, solver="lbfgs", max_iter=5000, random_state=SEED,
        ),
    }


# ==================== 3. KJØR FORSØKENE ====================

def run_experiments(X, y, splits):
    """Prøv alle PCA-valg og modeller på de samme foldene."""
    results = []

    # Kryssvalidering: én del brukes til validering, resten til trening.
    # Hvert bilde får være i valideringsdelen én gang.
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        for n_components in COMPONENTS:
            for whiten in WHITENING:
                # PCA lærer bare fra denne foldens treningsbilder.
                # whiten=True skalerer komponentene til lik varians.
                pca = PCA(
                    n_components=n_components,
                    whiten=whiten,
                    svd_solver="full",
                    random_state=SEED,
                )
                X_train_pca = pca.fit_transform(X_train)
                X_validation_pca = pca.transform(X_validation)
                # Behold samme tallpresisjon som den tidligere analysen.
                if not whiten:
                    X_train_pca = X_train_pca.astype(np.float64)
                    X_validation_pca = X_validation_pca.astype(np.float64)
                explained_variance = float(pca.explained_variance_ratio_.sum())

                # Alle fire algoritmer får nøyaktig de samme PCA-dataene.
                for name, model in classifiers().items():
                    model.fit(X_train_pca, y_train)
                    predictions = model.predict(X_validation_pca)

                    # Accuracy: andelen bilder med riktig personnummer.
                    # Macro F1: beregn F1 per person og gi hver person lik vekt.
                    accuracy = accuracy_score(y_validation, predictions)
                    macro_f1 = f1_score(
                        y_validation, predictions, average="macro", zero_division=0,
                    )
                    results.append({
                        "fold": fold,
                        "model": name,
                        "n_components": n_components,
                        "whiten": whiten,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "explained_variance": explained_variance,
                    })

        print(f"Ferdig med fold {fold} av {len(splits)}", flush=True)

    return results


def summarize_results(results):
    """Samle foldene til ett gjennomsnitt per modell og PCA-valg."""
    summary = []
    for name in classifiers():
        for n_components in COMPONENTS:
            for whiten in WHITENING:
                matching_rows = []
                for row in results:
                    if (row["model"] == name
                            and row["n_components"] == n_components
                            and row["whiten"] == whiten):
                        matching_rows.append(row)

                average = {
                    "model": name,
                    "n_components": n_components,
                    "whiten": whiten,
                }
                for metric in ["accuracy", "macro_f1", "explained_variance"]:
                    values = [row[metric] for row in matching_rows]
                    average[metric + "_mean"] = float(np.mean(values))
                    # Standardavvik viser hvor mye resultatene varierer mellom foldene.
                    average[metric + "_sd"] = float(np.std(values, ddof=1))
                summary.append(average)

    return summary


def rank(result):
    """Velg høyest accuracy, deretter F1, færre komponenter og uten whitening."""
    return (
        result["accuracy_mean"],
        result["macro_f1_mean"],
        -result["n_components"],
        not result["whiten"],
    )


def best_per_model(summary):
    best = []
    for name in classifiers():
        candidates = [row for row in summary if row["model"] == name]
        best.append(max(candidates, key=rank))
    return best


# ==================== 4. LAG GRAFER ====================

def save_figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pca_comparison(summary, figures):
    """Vis hvordan komponentantall og whitening påvirker RBF-SVM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for whiten in WHITENING:
        rows = []
        for row in summary:
            if row["model"] == "SVM RBF" and row["whiten"] == whiten:
                rows.append(row)
        axes[0].errorbar(
            COMPONENTS,
            [row["accuracy_mean"] for row in rows],
            yerr=[row["accuracy_sd"] for row in rows],
            marker="o", capsize=3, label=f"whiten={whiten}",
        )

    # Whitening endrer skalaen, men ikke hvor mye varians PCA forklarer.
    axes[1].errorbar(
        COMPONENTS,
        [row["explained_variance_mean"] for row in rows],
        yerr=[row["explained_variance_sd"] for row in rows],
        marker="o", capsize=3,
    )
    for ax in axes:
        ax.set_xlabel("Antall PCA-komponenter")
        ax.set_xticks(COMPONENTS)
        ax.grid(alpha=0.25)
    axes[0].set(title="RBF-SVM: gjennomsnitt ± standardavvik", ylabel="Accuracy")
    axes[0].legend()
    axes[1].set(title="PCA trent separat i hver fold", ylabel="Samlet forklart varians")
    save_figure(fig, figures, "pca_accuracy_variance")


def plot_algorithm_comparison(summary, figures):
    """Vis hver algoritme med sitt beste undersøkte PCA-valg."""
    best = best_per_model(summary)
    positions = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        positions - 0.18, [row["accuracy_mean"] for row in best], width=0.36,
        yerr=[row["accuracy_sd"] for row in best], capsize=3, label="Accuracy",
    )
    ax.bar(
        positions + 0.18, [row["macro_f1_mean"] for row in best], width=0.36,
        yerr=[row["macro_f1_sd"] for row in best], capsize=3, label="Macro F1",
    )
    labels = []
    for row in best:
        labels.append(
            f"{row['model']}\nk={row['n_components']}, whiten={row['whiten']}"
        )
    ax.set_xticks(positions, labels)
    ax.set(
        ylabel="Valideringsresultat", ylim=(0, 1.05),
        title="Beste PCA-valg per algoritme (gjennomsnitt ± standardavvik)",
    )
    ax.legend()
    save_figure(fig, figures, "algorithm_comparison")


def plot_eigenfaces(X_train, figures):
    """Vis PCA-mønstrene som bilder. Bare treningsdata brukes."""
    pca = PCA(n_components=10, svd_solver="full")
    pca.fit(X_train)
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    colour_limit = np.abs(pca.components_).max()
    for i, ax in enumerate(axes.flat):
        eigenface = pca.components_[i].reshape(64, 64)
        ax.imshow(eigenface, cmap="RdBu_r", vmin=-colour_limit, vmax=colour_limit)
        ax.set_title(f"PC {i + 1}")
        ax.axis("off")
    fig.suptitle("De første 10 eigenfaces – bare treningsdata, felles fargeskala")
    save_figure(fig, figures, "eigenfaces")


def plot_svm_illustration(X_train, y_train, figures, people):
    """Tren en egen liten modell for å kunne tegne grensen i to dimensjoner."""
    selected = np.isin(y_train, people)
    pca = PCA(n_components=2, whiten=True, svd_solver="full")
    points = pca.fit_transform(X_train[selected])
    labels = y_train[selected]
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(points, labels)

    # Lag et rutenett og spør SVM hvilken side av grensen hvert punkt ligger på.
    low = points.min(axis=0) - 0.8
    high = points.max(axis=0) + 0.8
    xx, yy = np.meshgrid(
        np.linspace(low[0], high[0], 350),
        np.linspace(low[1], high[1], 350),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    decision = svm.decision_function(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(
        xx, yy, decision > 0, levels=[-0.5, 0.5, 1.5],
        colors=["#dbe9f6", "#ffe4ce"], alpha=0.8,
    )
    contours = ax.contour(
        xx, yy, decision, levels=[-1, 0, 1],
        colors="black", linestyles=["--", "-", "--"],
    )
    ax.clabel(contours, fmt={-1: "f=-1", 0: "Grense", 1: "f=1"})
    for label, colour in zip(people, ["#2365a4", "#b75608"]):
        person_points = points[labels == label]
        ax.scatter(
            person_points[:, 0], person_points[:, 1],
            label=f"Person {label} (trening)", c=colour, edgecolors="white", s=40,
        )
    ax.scatter(
        svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
        facecolors="none", edgecolors="black", s=130, label="Support vectors",
    )
    ax.set(
        xlabel="Whitened PC 1", ylabel="Whitened PC 2",
        title="Separat 2D-illustrasjon: RBF-SVM, C=10\n"
              "Dette er ikke grensen til den 100-dimensjonale modellen",
    )
    ax.legend(fontsize=9)
    save_figure(fig, figures, "svm_2d_illustration")


# ==================== 5. LAGRE RESULTATENE ====================

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary, output):
    """Skriv en lesbar oppsummering basert på de faktiske resultatene."""
    best = max(summary, key=rank)
    lines = [
        "# PCA- og algoritmeanalyse", "",
        "Femfolds kryssvalidering som standard, bare på treningssettet. "
        "Se metadata.json for antall folder i denne kjøringen. Testsettet er ikke evaluert.", "",
        "| Modell | PCA | Whitening | Accuracy ± SD | Macro F1 ± SD |",
        "|---|---:|---|---:|---:|",
    ]
    for row in best_per_model(summary):
        lines.append(
            f"| {row['model']} | {row['n_components']} | {row['whiten']} "
            f"| {row['accuracy_mean']:.4f} ± {row['accuracy_sd']:.4f} "
            f"| {row['macro_f1_mean']:.4f} ± {row['macro_f1_sd']:.4f} |"
        )
    lines.extend([
        "", f"Beste observerte konfigurasjon: **{best['model']}, "
        f"{best['n_components']} komponenter, whiten={best['whiten']}**.",
        "Valgt etter accuracy, deretter macro F1, færre komponenter og uten whitening ved likhet.",
        "", "## RBF-SVM og whitening", "",
        "| PCA | Accuracy uten whitening | Accuracy med whitening | Differanse (med − uten) | Forklart varians |",
        "|---:|---:|---:|---:|---:|",
    ])
    for n_components in COMPONENTS:
        for row in summary:
            if row["model"] == "SVM RBF" and row["n_components"] == n_components:
                if row["whiten"]:
                    with_whitening = row
                else:
                    without_whitening = row
        difference = with_whitening["accuracy_mean"] - without_whitening["accuracy_mean"]
        lines.append(
            f"| {n_components} | {without_whitening['accuracy_mean']:.4f} "
            f"| {with_whitening['accuracy_mean']:.4f} | {difference:+.4f} "
            f"| {without_whitening['explained_variance_mean']:.4f} |"
        )

    best_models = best_per_model(summary)
    rbf = next(row for row in best_models if row["model"] == "SVM RBF")
    linear = next(row for row in best_models if row["model"] == "SVM Linear")
    accuracy_difference = 100 * (rbf["accuracy_mean"] - linear["accuracy_mean"])
    f1_difference = rbf["macro_f1_mean"] - linear["macro_f1_mean"]
    lines.extend([
        "", f"RBF-SVM minus lineær SVM ved hvert sitt beste PCA-valg: "
        f"{accuracy_difference:+.3f} prosentpoeng accuracy og {f1_difference:+.4f} macro F1.",
        "", "## Hva betyr resultatene?", "",
        "Mer forklart pikselvarians betyr ikke automatisk bedre gjenkjenning. "
        "Flere komponenter kan beskrive lys, bakgrunn og støy. Whitening gir også "
        "lavvarianskomponenter lik vekt og kan forsterke støy. Dette er mulige "
        "forklaringer, ikke årsaker bevist av forsøkene. gamma='scale' endres "
        "også med komponentantall og whitening.", "",
        "SVM bruker C=10. C og gamma kan tunes senere på treningssettet. "
        "KNN bruker k=5 og logistisk regresjon C=1. Dette er en sammenligning "
        "med faste parametere, ikke full tuning av hver algoritme.", "",
        "SD er standardavvik mellom foldene, ikke et konfidensintervall. Små "
        "forskjeller dokumenterer ikke statistisk overlegenhet. Beste modell "
        "er valgt på samme CV og kan derfor ha et optimistisk resultat. "
        "Etter endelig modellvalg må ytelsen måles én gang på testsettet, "
        "eventuelt med nested CV før dette.", "",
        "Private klasser har flere bilder enn hver Olivetti-person. Accuracy "
        "gir hvert bilde lik vekt; macro F1 gir hver person lik vekt. Bilder "
        "fra samme kameraøkt kan være svært like selv etter fjerning av eksakte "
        "duplikater. Gruppebasert CV eller en ny kameraøkt er strengere.", "",
        "Vi bruker predict(), uten appens predict_proba/decide og 50 %-avvisning. "
        "Resultatene måler derfor ikke appens avvisningsregel. "
        "Høyeste sannsynlighet og predict() kan gi ulike klasseforslag.", "",
        "Eigenfaces er laget bare fra treningssettet. SVM-figuren er en "
        "separat illustrasjonsmodell med to personer og to komponenter, "
        "ikke beslutningsgrensen til den 100-dimensjonale modellen.",
    ])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_folders():
    """Bruk faste mapper. Flytt forrige analyse til arkivet før en ny lagres."""
    output = ROOT / "reports" / "model_analysis"
    figures = ROOT / "reports" / "figures" / "model_analysis"
    if output.exists() or figures.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = ROOT / "reports" / "archive" / f"comparison_{timestamp}"
        archive.mkdir(parents=True, exist_ok=False)
        if output.exists():
            output.rename(archive / "results")
        if figures.exists():
            figures.rename(archive / "figures")
    output.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    return output, figures


def save_metadata(X, y, counts, splits, people, output):
    """Ta vare på innstillinger og foldindekser så forsøket kan gjentas."""
    parameters = {}
    for name, model in classifiers().items():
        parameters[name] = model.get_params()
    metadata = {
        "seed": SEED,
        "split_seed": 30,
        "folds": len(splits),
        "training_shape": list(X.shape),
        "training_class_counts": counts,
        "training_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "components": COMPONENTS,
        "illustration_labels": people,
        "pca_solver": "full",
        "classifier_parameters": parameters,
        "selection": "mean accuracy, macro F1, fewer components, no whitening",
        "test_evaluated": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    fold_indices = {}
    for i, (training, validation) in enumerate(splits):
        fold_indices[f"train_{i}"] = training
        fold_indices[f"validation_{i}"] = validation
    np.savez(output / "cv_indices.npz", **fold_indices)




if __name__ == "__main__":
    main()
