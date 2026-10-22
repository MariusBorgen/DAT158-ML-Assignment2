"""Kjør denne filen for å vise grafene fra siste trening."""
import numpy as np
import matplotlib.pyplot as plt
from face_utils import ROOT, PERSON_LABELS, NAMES

figure_dir = ROOT / "reports" / "figures"

def person_name(label):
    return NAMES.get(label, f"Person {label}")

def save_and_show(filename):
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_results(X_test, y_test, predictions, correct_counts, wrong_counts,
                 rejected_counts, train_accuracy, test_accuracy, threshold):
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Riktige navn, feil navn og avviste treff per person.
    positions = np.arange(len(PERSON_LABELS))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(positions - 0.25, correct_counts, width=0.25,
                  label="Riktig navn", color="seagreen")
    ax.bar_label(bars, padding=3)
    bars = ax.bar(positions, wrong_counts, width=0.25,
                  label="Feil navn", color="firebrick")
    ax.bar_label(bars, padding=3)
    bars = ax.bar(positions + 0.25, rejected_counts, width=0.25,
                  label="Avvist", color="darkorange")
    ax.bar_label(bars, padding=3)
    ax.set_xticks(positions, list(PERSON_LABELS))
    ax.set_title(f"Resultater per person – grense over {threshold:.0%}")
    ax.set_ylabel("Antall testbilder")
    ax.set_ylim(0, max(correct_counts + wrong_counts + rejected_counts) + 6)
    ax.legend()
    save_and_show("01_treffsikkerhet_per_person.png")

    # 7. Vis feilklassifiserte testbilder.
    wrong_indices = []
    for i in range(len(y_test)):
        if predictions[i] != y_test[i]:
            wrong_indices.append(i)

    if len(wrong_indices) > 0:
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

        fig.suptitle("Feilklassifiserte testbilder – før avvisning")
        save_and_show("02_feilklassifiserte_bilder.png")

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["Treningsbilder", "Testbilder"],
        [train_accuracy * 100, test_accuracy * 100],
        color=["steelblue", "darkorange"],
    )

    ax.bar_label(bars, fmt="%.1f%%", padding=4)

    ax.set(
        title="Treffsikkerhet på trening og test – før avvisning",
        ylabel="Treffsikkerhet (%)",
        ylim=(0, 110),
    )

    save_and_show("03_trening_mot_test.png")

    print(f"Grafene er lagret i {figure_dir}")


if __name__ == "__main__":
    result_path = ROOT / "reports" / "training_results.npz"
    if not result_path.exists():
        raise SystemExit("Kjør python src/train.py først for å lagre resultater.")

    # Les resultatene fra siste trening uten å trene modellen på nytt.
    with np.load(result_path) as results:
        plot_results(
            results["X_test"], results["y_test"], results["predictions"],
            results["correct_counts"].tolist(),
            results["wrong_counts"].tolist(),
            results["rejected_counts"].tolist(),
            float(results["train_accuracy"]), float(results["test_accuracy"]),
            float(results["threshold"]),
        )
