"""Evaluate AntiNODE reproduction results (CIFAR-10, Dopri5).

Reads the merged per-image attack results (5 betas), prints a paper-style
comparison table (NFE, L2, wall-clock latency), generates three figures
(NFE distribution, NFE vs L2 scatter, attack success rates), and saves
an aggregated summary JSON.

Wall-clock latency is measured for the benign forward pass and for each
adversarial beta whose tensors were saved by attack.py in
`results/adv_tensors/beta_{beta}.pt`. Protocol: batch=1, warmup forwards,
cuda.synchronize + perf_counter bracketing each timed forward.

Beta columns:
  - beta=0      (unrestricted)
  - beta=0.001  (restricted weak)
  - beta=0.01   (restricted strong, matches paper's operating regime)
  - beta=0.1    (beyond paper; kept in summary.json only)
  - beta=1      (beyond paper; kept in summary.json only)
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from attack import normalize_cifar10
from model import ODEClassifier

CHECKPOINT_PATH = Path("checkpoints/best.pth")
DATA_DIR = Path("data")
LATENCY_WARMUP = 10

# Setting key → adversarial beta (float). Keep in sync with load_results()
# extraction keys and with attack.py's `beta_{beta}.pt` filename format.
BETA_FOR_SETTING = {
    "unrestricted_0": 0.0,
    "restricted_0.001": 0.001,
    "restricted_0.01": 0.01,
    "restricted_0.1": 0.1,
    "restricted_1": 1.0,
}
ADV_SETTINGS = list(BETA_FOR_SETTING.keys())

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

def load_results(json_path):
    """Return {setting_label: records} with records as list of dicts.

    Settings: 'benign' plus the 5 keys in BETA_FOR_SETTING. Each record:
    {'nfe', 'l2', 'orig_pred', 'adv_pred'}. Benign records have
    l2=None and adv_pred=None.
    """
    with open(json_path) as f:
        data = json.load(f)

    benign = [
        {"nfe": r["orig_nfe"], "l2": None,
         "orig_pred": r["orig_pred"], "adv_pred": None}
        for r in data
    ]
    settings = {"benign": benign}
    for key, beta in BETA_FOR_SETTING.items():
        # attack.py stores beta keys as str(float): "0.0", "0.001", "1.0" …
        settings[key] = _extract_beta(data, str(beta))
    return settings


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


# ── Wall-clock latency ───────────────────────────────────────────────

def _load_model(device):
    model = ODEClassifier().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _time_forward_passes(model, device, images):
    """Time per-image forward passes. `images` must be already-normalized
    tensors of shape [1,3,32,32] already on device.
    """
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            model(images[0])
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms = []
        for x in images:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "n": len(images),
        "device": device.type,
    }


def measure_latencies(num_images, adv_tensors_dir):
    """Measure benign + adversarial forward-pass latency.

    Benign: first num_images from CIFAR-10 test set.
    Adversarial: loaded from `adv_tensors_dir/beta_{beta}.pt` for each
    setting in BETA_FOR_SETTING. Settings with no saved tensor file are
    skipped (warning printed).

    Returns {"benign": stats, "adv": {setting_key: stats}}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)

    test_set = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=False,
        transform=transforms.ToTensor(),
    )
    benign_images = [
        normalize_cifar10(test_set[i][0].unsqueeze(0).to(device))
        for i in range(num_images)
    ]
    benign_stats = _time_forward_passes(model, device, benign_images)

    adv_stats = {}
    for key, beta in BETA_FOR_SETTING.items():
        path = adv_tensors_dir / f"beta_{beta}.pt"
        if not path.exists():
            print(f"  skip adv latency for {key}: {path} not found")
            continue
        blob = torch.load(path, map_location="cpu", weights_only=True)
        imgs = blob["images"]  # [N, 3, 32, 32] in [0, 1]
        adv_images = [
            normalize_cifar10(imgs[i].unsqueeze(0).to(device))
            for i in range(imgs.shape[0])
        ]
        adv_stats[key] = _time_forward_passes(model, device, adv_images)

    return {"benign": benign_stats, "adv": adv_stats}


# ── Table printing ───────────────────────────────────────────────────

def _fmt_latency(stats):
    if stats is None:
        return "-"
    return f"{stats['mean_ms']:.2f}±{stats['std_ms']:.2f}"


def print_reproduction_table(settings, latency_data):
    n = len(settings["benign"])
    benign_nfe = [r["nfe"] for r in settings["benign"]]
    benign_mean = float(np.mean(benign_nfe))
    benign_lat = latency_data["benign"]

    rows = [("Seed (benign)", benign_nfe, None, None, _fmt_latency(benign_lat))]
    labels = {
        "unrestricted_0": "Unrestricted beta=0",
        "restricted_0.001": "Restricted beta=0.001",
        "restricted_0.01": "Restricted beta=0.01",
        "restricted_0.1": "Restricted beta=0.1",
        "restricted_1": "Restricted beta=1",
    }
    for key, label in labels.items():
        nfes = [r["nfe"] for r in settings[key]]
        l2s = [r["l2"] for r in settings[key]]
        lat_str = _fmt_latency(latency_data["adv"].get(key))
        rows.append((label, nfes, l2s, benign_mean, lat_str))

    print(f"\nAntiNODE Reproduction - CIFAR-10, Dopri5, {n} test images\n")
    header = (
        f"{'Attack':<24}{'NFE (mean)':>12}{'Increase':>12}"
        f"{'Iters':>10}{'L2 (mean)':>12}{'Latency (ms)':>16}"
    )
    print(header)
    print("-" * len(header))

    for label, nfes, l2s, baseline, latency_str in rows:
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
            f"{iters_equiv:>9.1f} {l2_str:>12}{latency_str:>16}"
        )

    print(
        f"\nLatency: batch=1, device={benign_lat['device']}, "
        f"warmup={LATENCY_WARMUP}. '-' = adv tensors not found on disk."
    )
    print(f"\nPaper reference (Table 1, CIFAR-10 Dopri5):")
    print(f"  Unrestricted: {PAPER_UNRESTRICTED_ITERS} iters (+{PAPER_UNRESTRICTED_INCREASE}%)")
    print(f"  Restricted:   {PAPER_RESTRICTED_ITERS} iters (+{PAPER_RESTRICTED_INCREASE}%)")


# ── Plots ────────────────────────────────────────────────────────────

# Figures stay on the 3 paper-aligned betas. Adding beta=0.1 and beta=1
# to the NFE histogram oversaturates it with near-baseline curves, and
# on the NFE-vs-L2 scatter their tiny L2 range compresses the interesting
# regime of beta={0, 0.001, 0.01}. The full 5-beta sweep lives in
# summary.json; plots are only for the regime that matches Table 1.
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

def write_summary_json(settings, latency_data, path):
    benign_nfe = [r["nfe"] for r in settings["benign"]]

    out = {
        "num_images": len(settings["benign"]),
        "nfe_per_iter": NFE_PER_ITER,
        "settings": {},
    }
    out["settings"]["benign"] = {
        "nfe": summary_stats(benign_nfe),
        "latency_ms": latency_data["benign"],
    }

    for key in ADV_SETTINGS:
        recs = settings[key]
        nfes = [r["nfe"] for r in recs]
        l2s = [r["l2"] for r in recs]
        setting_out = {
            "nfe": summary_stats(nfes),
            "l2": summary_stats(l2s),
            "attack_success_rate": attack_success_rate(recs, benign_nfe),
            "label_flip_rate": label_flip_rate(recs),
        }
        if key in latency_data["adv"]:
            setting_out["latency_ms"] = latency_data["adv"][key]
        out["settings"][key] = setting_out

    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────

def main(args):
    results_dir = Path(args.results_dir)
    results_json = results_dir / "attack_results.json"
    figures_dir = results_dir / "figures"
    summary_path = results_dir / "summary.json"
    adv_tensors_dir = results_dir / "adv_tensors"

    figures_dir.mkdir(parents=True, exist_ok=True)
    settings = load_results(results_json)
    latency_data = measure_latencies(len(settings["benign"]), adv_tensors_dir)
    print_reproduction_table(settings, latency_data)
    plot_nfe_distribution(settings, figures_dir / "nfe_distribution.pdf")
    plot_nfe_vs_l2(settings, figures_dir / "nfe_vs_l2.pdf")
    plot_attack_success_rate(settings, figures_dir / "attack_success_rate.pdf")
    write_summary_json(settings, latency_data, summary_path)
    print(f"\nSaved: {figures_dir}/*.pdf and {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AntiNODE results")
    parser.add_argument("--results-dir", type=str, default="results")
    main(parser.parse_args())
