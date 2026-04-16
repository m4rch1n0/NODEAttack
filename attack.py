"""AntiNODE latency-surging attack on a Neural ODE classifier (CIFAR-10).

Implements the C&W-style input perturbation from AntiNODE (ICCVW 2023):
optimizes an adversarial image to minimize the adaptive solver's initial
step size, forcing more integration steps (higher NFE = higher latency).

The differentiable proxy is the initial step size computed by the HNW
heuristic (Hairer-Norsett-Wanner). This requires only 2 ODE function
evaluations per optimization step, making backpropagation fast.

NFE is measured separately using torchdiffeq's standard odeint.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from model import ODEClassifier

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# ── Differentiable step-size proxy ───────────────────────────────────

def _rms_norm(x):
    return x.norm() / (x.numel() ** 0.5)


def initial_step_size(func, y0, t0=0.0, rtol=1e-3, atol=1e-3):
    """Compute the initial step size that dopri5 would select (HNW heuristic).

    This is differentiable w.r.t. y0 because it depends on:
      - ||y0|| (initial state)
      - ||f(t0, y0)|| (first function evaluation)
      - ||(f(t0+h0, y0+h0*f0) - f0)/h0|| (curvature estimate)

    Only 2 ODE function evaluations — fast to backpropagate through.

    Reference: Hairer, Norsett, Wanner, "Solving ODEs I", Sec. II.4.
    """
    dtype, device = y0.dtype, y0.device
    if not torch.is_tensor(t0):
        t0 = torch.tensor(t0, dtype=dtype, device=device)

    scale = atol + torch.abs(y0) * rtol
    f0 = func(t0, y0)
    d0 = _rms_norm(y0 / scale)
    d1 = _rms_norm(f0 / scale)

    if d0.item() < 1e-5 or d1.item() < 1e-5:
        h0 = torch.tensor(1e-6, dtype=dtype, device=device)
    else:
        h0 = 0.01 * d0 / d1

    # Second evaluation: estimate curvature
    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)
    d2 = _rms_norm((f1 - f0) / scale) / h0

    if d1.item() <= 1e-15 and d2.item() <= 1e-15:
        h1 = torch.max(
            torch.tensor(1e-6, dtype=dtype, device=device), h0 * 1e-3,
        )
    else:
        h1 = (0.01 / torch.max(d1, d2)) ** (1.0 / 6.0)

    return torch.min(100.0 * h0, h1)


# ── Helpers ──────────────────────────────────────────────────────────

def normalize_cifar10(x):
    """Normalize a [0, 1] image tensor to CIFAR-10 statistics."""
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def measure_nfe(model, x_normalized, device):
    """Run a standard forward pass and return (predicted_label, nfe)."""
    with torch.no_grad():
        logits = model(x_normalized.to(device))
        return logits.argmax(dim=1).item(), model.nfe


# ── Attack ───────────────────────────────────────────────────────────

def attack_single(model, x_orig, beta, device, iters=2000, lr=5e-4):
    """Run the AntiNODE attack on a single image.

    Minimizes the initial step size of the dopri5 solver (differentiable
    proxy for NFE) while optionally penalizing L2 perturbation distance.

    Args:
        model: trained ODEClassifier (eval mode, parameters frozen).
        x_orig: clean image in [0, 1], shape [1, 3, 32, 32].
        beta: L2 penalty weight (0 = unrestricted).
        device: torch device.
        iters: number of Adam iterations.
        lr: learning rate.

    Returns:
        dict with best adversarial image, NFE, L2, etc.
    """
    x_orig_dev = x_orig.to(device)

    # Tanh reparametrization: optimize w in unconstrained space
    # x_adv = 0.5 * (tanh(w) + 1), so w = atanh(2*x - 1)
    w = torch.atanh((2 * x_orig_dev - 1).clamp(-1 + 1e-5, 1 - 1e-5))
    w = w.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    odefunc = model.ode_block.odefunc
    best_nfe = 0
    best_x_adv = x_orig_dev.clone()
    best_pred = -1

    for i in range(iters):
        optimizer.zero_grad()

        # Map back to image space [0, 1], then normalize
        x_adv = 0.5 * (torch.tanh(w) + 1)
        x_norm = normalize_cifar10(x_adv)

        # Forward through downsampling to get ODE initial condition
        features = model.downsampling(x_norm)

        # Differentiable proxy: initial step size (2 func evals, fast)
        h0 = initial_step_size(odefunc, features, t0=0.0, rtol=1e-3, atol=1e-3)

        # Loss: minimize h0 (smaller initial step = more solver iterations)
        l2_dist = ((x_adv - x_orig_dev) ** 2).sum()
        loss = h0 + beta * l2_dist

        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
        optimizer.step()

        # Periodically measure actual NFE with torchdiffeq
        if i % 100 == 0 or i == iters - 1:
            with torch.no_grad():
                x_eval = normalize_cifar10(0.5 * (torch.tanh(w) + 1))
                try:
                    logits = model(x_eval)
                    nfe = model.nfe
                    pred = logits.argmax(dim=1).item()
                except (AssertionError, RuntimeError):
                    # Solver underflow = dynamics too stiff to integrate
                    nfe = 9999
                    pred = -1

            if nfe > best_nfe:
                best_nfe = nfe
                best_x_adv = (0.5 * (torch.tanh(w) + 1)).detach()
                best_pred = pred

            if i % 500 == 0:
                nfe_str = "FAIL" if nfe == 9999 else str(nfe)
                print(
                    f"    iter {i:4d}: h0={h0.item():.6f}, "
                    f"L2={l2_dist.item():.4f}, NFE={nfe_str}"
                )

    # Final evaluation of the best adversarial example
    x_best_norm = normalize_cifar10(best_x_adv)
    try:
        best_pred_final, best_nfe_final = measure_nfe(model, x_best_norm, device)
    except (AssertionError, RuntimeError):
        best_pred_final, best_nfe_final = -1, 9999
    best_l2 = ((best_x_adv - x_orig_dev) ** 2).sum().item()

    return {
        "best_x_adv": best_x_adv,
        "best_nfe": best_nfe_final,
        "best_pred": best_pred_final,
        "best_l2": best_l2,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = ODEClassifier().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Freeze model parameters (we only optimize the perturbation)
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loaded checkpoint: {args.checkpoint} (acc={ckpt['accuracy']:.4f})")

    # Load test images (unnormalized, [0, 1])
    test_set = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True,
        transform=transforms.ToTensor(),
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    betas = [float(b) for b in args.betas.split(",")]
    all_results = []

    for idx in tqdm(range(args.num_images), desc="Images"):
        x_orig, label = test_set[idx]
        x_orig = x_orig.unsqueeze(0)

        # Measure baseline NFE
        x_norm = normalize_cifar10(x_orig.to(device))
        orig_pred, orig_nfe = measure_nfe(model, x_norm, device)

        img_result = {
            "index": idx,
            "label": label,
            "orig_pred": orig_pred,
            "orig_nfe": orig_nfe,
            "attacks": {},
        }

        print(f"\nImage {idx}: label={label}, pred={orig_pred}, baseline NFE={orig_nfe}")

        for beta in betas:
            print(f"  beta={beta}")
            t_start = time.time()

            result = attack_single(
                model, x_orig, beta, device,
                iters=args.iters, lr=args.lr,
            )

            elapsed = time.time() - t_start

            img_result["attacks"][str(beta)] = {
                "adv_nfe": result["best_nfe"],
                "adv_pred": result["best_pred"],
                "l2_dist": result["best_l2"],
                "time_s": round(elapsed, 1),
            }

            print(
                f"    -> NFE: {orig_nfe} -> {result['best_nfe']}, "
                f"L2={result['best_l2']:.4f}, time={elapsed:.1f}s"
            )

        all_results.append(img_result)

        # Save incrementally
        with open(results_dir / "attack_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for beta in betas:
        nfes_orig = [r["orig_nfe"] for r in all_results]
        nfes_adv = [r["attacks"][str(beta)]["adv_nfe"] for r in all_results]
        l2s = [r["attacks"][str(beta)]["l2_dist"] for r in all_results]
        print(
            f"  beta={beta:>7}: "
            f"NFE {sum(nfes_orig)/len(nfes_orig):.1f} -> {sum(nfes_adv)/len(nfes_adv):.1f}, "
            f"avg L2={sum(l2s)/len(l2s):.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AntiNODE latency attack on CIFAR-10")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--betas", type=str, default="0,0.001,0.01,0.1,1",
                        help="Comma-separated L2 penalty weights "
                             "(defaults are calibrated for our h0 proxy; "
                             "see lab notebook for the discrepancy vs paper)")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    main(parser.parse_args())
