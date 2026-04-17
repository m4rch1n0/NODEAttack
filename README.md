# NODEAttack — Reproduction

Clean PyTorch 2.11 reproduction of **AntiNODE** (ICCVW 2023), a
latency-surging adversarial attack against Neural ODE classifiers, on
CIFAR-10 with the Dopri5 adaptive solver. Implemented as the reproduction
step of a bachelor thesis on latency-surging attacks against generative
flow models.

- **Paper:** [AntiNODE project website](https://sites.google.com/view/nodeattack/home) (ICCVW 2023)
- **Upstream:** the original abandoned PyTorch ~1.0 code is preserved on
  branch [`main`](../../tree/main); this reproduction lives on
  [`reproduce-pytorch2`](../../tree/reproduce-pytorch2).

## Setup

```bash
uv sync                      # PyTorch 2.11+rocm7.2, torchdiffeq 0.2.5, etc.
uv run python train.py       # train ODEClassifier on CIFAR-10 (~50 epochs)
uv run python attack.py      # run the AntiNODE attack
uv run python evaluate.py    # NFE table, wall-clock latency, figures
```

Optimizer matches the paper (Adam, lr=5e-4, 2000 iterations, dopri5 with
rtol=atol=1e-3). The `--betas` default is `"0,0.001,0.01,0.1,1"`, which
reproduces the results table below; the paper's nominal β ∈ {10, 100,
1000, 10000} are calibrated for a different loss proxy and saturate at
baseline NFE on our stack — see the thesis lab notebook for the
β-scaling discussion.

Attack flags: `--checkpoint`, `--num-images`, `--betas`, `--iters`,
`--lr`, `--data-dir`, `--results-dir`.

## Results

CIFAR-10, Dopri5 (rtol=atol=1e-3), 250 test images, ODEClassifier
trained to 84.47% accuracy.

MNIST reproduction was out of scope: CIFAR-10 + Dopri5 is Table 1's
primary setting and the one this thesis's Phase 2 (Flow Matching) will
build on.

| Setting              | NFE mean | Increase | L2 mean | Success | Flip |
|----------------------|---------:|---------:|--------:|--------:|-----:|
| Benign               |   26.00  |      —   |    —    |    —    |  —   |
| Unrestricted β=0     |   34.18  | +31.5%   |  2.69   |  93.2%  | 18%  |
| Restricted β=0.001   |   33.39  | +28.4%   |  1.70   |  87.6%  | 17%  |
| Restricted β=0.01    |   29.00  | +11.5%   |  0.22   |  36.8%  |  8%  |

Paper reference (Table 1, CIFAR-10 Dopri5): Unrestricted +42.5%,
Restricted +37.5%. β ∈ {0.1, 1} extend beyond the paper's restricted
values; see `summary.json` for the full 5-β sweep.

Wall-clock latency is measured directly for benign and for each
adversarial β (batch=1, AMD RX 6900 XT + ROCm 7.2, 10 warmup forwards).
Benign baseline: **19.73 ± 0.40 ms** per image. Pooled across benign +
3 primary β (≈1000 points), per-image latency is a near-linear function
of NFE — **slope ≈ 0.69 ms/NFE, R² = 0.989** — which empirically
validates using NFE as a proxy for latency on this architecture. The
unrestricted attack's +31.5% NFE at β=0 therefore translates into
≈ +29% of inference time.

Figures and `summary.json` are not versioned in the repo. Regenerate
them with `uv run python evaluate.py` — outputs land in
`results/figures/` (NFE distribution, NFE vs L2 scatter, latency vs
NFE with linear fit, attack success / flip-rate bar charts) and
`results/summary.json`.

## Thesis context

This repository is the reproduction step of a bachelor thesis. The
follow-up work, which extends the latency-surging framework to Flow
Matching generative models, lives in the main thesis repository:

→ [m4rch1n0/slowflow](https://github.com/m4rch1n0/slowflow)

## License

MIT (inherited from the upstream torchdiffeq fork — see the `setup.py`
classifier on branch [`main`](../../tree/main)).