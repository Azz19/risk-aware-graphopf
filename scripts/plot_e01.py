from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    path = Path("results/E01_uncertainty")
    df = pd.read_csv(path / "summary.csv")
    labels = {
        "iid_gaussian": "IID Gaussian",
        "corr_gaussian": "Correlated Gaussian",
        "iid_student_t": "IID Student-t",
        "corr_student_t": "Correlated Student-t",
    }
    x = range(len(df))
    y = df["joint_violation_rate"]
    yerr = [y - df["ci_low"], df["ci_high"] - y]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    ax.set_xticks(list(x), [labels[v] for v in df["family"]], rotation=15, ha="right")
    ax.set_ylabel("Empirical joint violation probability")
    ax.set_xlabel("Renewable forecast-error model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path / "figure1_uncertainty_risk.pdf", bbox_inches="tight")
    fig.savefig(path / "figure1_uncertainty_risk.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
