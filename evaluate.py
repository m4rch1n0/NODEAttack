"""Evaluate AntiNODE reproduction results (CIFAR-10, Dopri5, 250 images).

Reads the merged per-image attack results (all betas across all runs),
extracts the useful beta columns, prints a paper-style comparison table,
generates three figures (NFE distribution, NFE vs L2 scatter, attack
success rates), and saves an aggregated summary JSON.

Useful beta columns (selected from all available):
  - beta=0      (unrestricted)
  - beta=0.001  (restricted weak)
  - beta=0.01   (restricted strong)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
RESULTS_JSON = RESULTS_DIR / "attack_results.json"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

# Dopri5 with FSAL does ~6.5 function evaluations per accepted step
# (7 for the first step, 6 for each subsequent step, plus 2 for the HNW
# initial-step heuristic; averaged over a typical 4-step solve this is ~6.5).
NFE_PER_ITER = 6.5

# Paper Table 1 reference (CIFAR-10, Dopri5)
PAPER_BASELINE_ITERS = 4.0
PAPER_UNRESTRICTED_ITERS = 5.7
PAPER_UNRESTRICTED_INCREASE = 42.5  # percent
PAPER_RESTRICTED_ITERS = 5.5
PAPER_RESTRICTED_INCREASE = 37.5  # percent


# ── Data loading ─────────────────────────────────────────────────────

def load_results():
    """Return {setting_label: records} with records as list of dicts.

    Settings:
        'benign'            — baseline NFE (no attack)
        'unrestricted_0'    — beta=0
        'restricted_0.001'  — beta=0.001
        'restricted_0.01'   — beta=0.01

    Each record: {'nfe', 'l2', 'orig_pred', 'adv_pred'}.
    Benign records have l2=None and adv_pred=None.
    """
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    benign = [
        {"nfe": r["orig_nfe"], "l2": None,
         "orig_pred": r["orig_pred"], "adv_pred": None}
        for r in data
    ]
    return {
        "benign": benign,
        "unrestricted_0": _extract_beta(data, "0.0"),
        "restricted_0.001": _extract_beta(data, "0.001"),
        "restricted_0.01": _extract_beta(data, "0.01"),
    }


def _extract_beta(records, beta_key):
    out = []
    for r in records:
        a = r["attacks"][beta_key]
        out.append({
            "nfe": a["adv_nfe"],
            "l2": a["l2_dist"],
            "orig_pred": r["orig_pred"],
            "adv_pred": a["adv_pred"],
        })
    return out


# ── Aggregation ──────────────────────────────────────────────────────

def summary_stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def attack_success_rate(records, benign_nfe):
    """Fraction of images where adv NFE strictly exceeds baseline NFE."""
    return sum(
        1 for r, b in zip(records, benign_nfe) if r["nfe"] > b
    ) / len(records)


def label_flip_rate(records):
    """Fraction of images where the predicted label changed under attack."""
    return sum(
        1 for r in records if r["adv_pred"] != r["orig_pred"]
    ) / len(records)


# ── Table printing ───────────────────────────────────────────────────

def print_reproduction_table(settings):
    n = len(settings["benign"])
    benign_nfe = [r["nfe"] for r in settings["benign"]]
    benign_mean = float(np.mean(benign_nfe))

    rows = [("Seed (benign)", benign_nfe, None, None)]
    labels = {
        "unrestricted_0": "Unrestricted beta=0",
        "restricted_0.001": "Restricted beta=0.001",
        "restricted_0.01": "Restricted beta=0.01",
    }
    for key, label in labels.items():
        nfes = [r["nfe"] for r in settings[key]]
        l2s = [r["l2"] for r in settings[key]]
        rows.append((label, nfes, l2s, benign_mean))

    print(f"\nAntiNODE Reproduction - CIFAR-10, Dopri5, {n} test images\n")
    header = f"{'Attack':<24}{'NFE (mean)':>12}{'Increase':>12}{'Iters':>10}{'L2 (mean)':>12}"
    print(header)
    print("-" * len(header))

    for label, nfes, l2s, baseline in rows:
        mean_nfe = float(np.mean(nfes))
        iters_equiv = mean_nfe / NFE_PER_ITER
        if baseline is None:
            increase = "-"
            l2_str = "-"
        else:
            increase = f"+{(mean_nfe / baseline - 1) * 100:.1f}%"
            l2_str = f"{float(np.mean(l2s)):.2f}"
        print(
            f"{label:<24}{mean_nfe:>12.1f}{increase:>12}"
            f"{iters_equiv:>9.1f} {l2_str:>12}"
        )

    print(f"\nPaper reference (Table 1, CIFAR-10 Dopri5):")
    print(f"  Unrestricted: {PAPER_UNRESTRICTED_ITERS} iters (+{PAPER_UNRESTRICTED_INCREASE}%)")
    print(f"  Restricted:   {PAPER_RESTRICTED_ITERS} iters (+{PAPER_RESTRICTED_INCREASE}%)")


# ── Plots ────────────────────────────────────────────────────────────

PLOT_ORDER = ["benign", "unrestricted_0", "restricted_0.001", "restricted_0.01"]
PLOT_LABELS = {
    "benign": "Benign",
    "unrestricted_0": r"Unrestricted ($\beta=0$)",
    "restricted_0.001": r"Restricted ($\beta=0.001$)",
    "restricted_0.01": r"Restricted ($\beta=0.01$)",
}
PLOT_COLORS = {
    "benign": "#4c72b0",
    "unrestricted_0": "#dd8452",
    "restricted_0.001": "#55a868",
    "restricted_0.01": "#c44e52",
}


def plot_nfe_distribution(settings, out_path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    all_nfes = np.concatenate([
        [r["nfe"] for r in settings[k]] for k in PLOT_ORDER
    ])
    bins = np.arange(all_nfes.min() - 1, all_nfes.max() + 2) - 0.5

    for key in PLOT_ORDER:
        nfes = [r["nfe"] for r in settings[key]]
        ax.hist(
            nfes, bins=bins, alpha=0.5,
            color=PLOT_COLORS[key], label=PLOT_LABELS[key],
            edgecolor="black", linewidth=0.3,
        )
        ax.axvline(
            float(np.mean(nfes)), color=PLOT_COLORS[key],
            linestyle="--", linewidth=1.2,
        )

    ax.set_xlabel("NFE (number of function evaluations)")
    ax.set_ylabel("Count")
    ax.set_title("NFE distribution under latency attack")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_nfe_vs_l2(settings, out_path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    baseline_nfe = float(np.mean([r["nfe"] for r in settings["benign"]]))

    for key in ("unrestricted_0", "restricted_0.001", "restricted_0.01"):
        records = settings[key]
        l2s = [r["l2"] for r in records]
        nfes = [r["nfe"] for r in records]
        ax.scatter(
            l2s, nfes, s=12, alpha=0.55,
            color=PLOT_COLORS[key], label=PLOT_LABELS[key],
            edgecolors="none",
        )

    ax.axhline(
        baseline_nfe, color="gray", linestyle=":", linewidth=1.0,
        label=f"Baseline NFE = {baseline_nfe:.1f}",
    )
    ax.set_xlabel(r"$L_2$ distortion $\|x_{adv} - x\|_2^2$")
    ax.set_ylabel("Adversarial NFE")
    ax.set_title(r"NFE vs $L_2$ distortion")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_attack_success_rate(settings, out_path):
    keys = ["unrestricted_0", "restricted_0.001", "restricted_0.01"]
    benign_nfe = [r["nfe"] for r in settings["benign"]]

    success = [attack_success_rate(settings[k], benign_nfe) * 100 for k in keys]
    flips = [label_flip_rate(settings[k]) * 100 for k in keys]

    x = np.arange(len(keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(x - width / 2, success, width, label="Efficiency attack success",
           color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, flips, width, label="Label flip rate",
           color="#dd8452", edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([PLOT_LABELS[k] for k in keys], fontsize=9)
    ax.set_ylabel("Percentage of 250 images")
    ax.set_title("Efficiency attack vs accuracy attack")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Summary JSON ─────────────────────────────────────────────────────

def write_summary_json(settings, path):
    benign_nfe = [r["nfe"] for r in settings["benign"]]

    out = {
        "num_images": len(settings["benign"]),
        "nfe_per_iter": NFE_PER_ITER,
        "settings": {},
    }
    out["settings"]["benign"] = {"nfe": summary_stats(benign_nfe)}

    for key in ("unrestricted_0", "restricted_0.001", "restricted_0.01"):
        recs = settings[key]
        nfes = [r["nfe"] for r in recs]
        l2s = [r["l2"] for r in recs]
        out["settings"][key] = {
            "nfe": summary_stats(nfes),
            "l2": summary_stats(l2s),
            "attack_success_rate": attack_success_rate(recs, benign_nfe),
            "label_flip_rate": label_flip_rate(recs),
        }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_results()
    print_reproduction_table(settings)
    plot_nfe_distribution(settings, FIGURES_DIR / "nfe_distribution.pdf")
    plot_nfe_vs_l2(settings, FIGURES_DIR / "nfe_vs_l2.pdf")
    plot_attack_success_rate(settings, FIGURES_DIR / "attack_success_rate.pdf")
    write_summary_json(settings, SUMMARY_PATH)
    print(f"\nSaved: {FIGURES_DIR}/*.pdf and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
