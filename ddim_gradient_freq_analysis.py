"""
실험: Pixel-space gradient의 저주파 편향 시각화
---------------------------------------------------
1. Jacobian ∂x0/∂xT 의 singular value spectrum
   - DDIM 내부가 no_grad이므로 finite differences로 Jv 계산
   - randomized SVD: Y = JΩ (k forward passes) → SVD(Y)

2. ∇xT L 의 FFT 분석
   - 매 iteration latent.grad 의 radial 주파수 스펙트럼 저장

3. 주파수 대역별 ℓ₂ norm 추이
   - low / mid / high band 로 분리하여 iteration별 플롯
"""

import os
import csv
import numpy as np
import torch
from torch import nn
import yaml
import sys
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guided_diffusion.models import Model
from ddim_inversion_utils import *
from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Config ────────────────────────────────────────────────────────────────────
with open('configs/blind_deblurring.yml', 'r') as f:
    task_config = yaml.safe_load(f)

torch.set_printoptions(sci_mode=False)
ensure_reproducibility(task_config['seed'])

with open('data/celeba_hq.yml', 'r') as f:
    config1 = yaml.safe_load(f)
config = dict2namespace(config1)
model, device = load_pretrained_diffusion_model(config)

ddim_scheduler = DDIMScheduler(
    beta_start=config.diffusion.beta_start,
    beta_end=config.diffusion.beta_end,
    beta_schedule=config.diffusion.beta_schedule,
)
ddim_scheduler.set_timesteps(
    config.diffusion.num_diffusion_timesteps // task_config['delta_t']
)

l2_loss = nn.MSELoss()
fixed_kernel_np = np.load('data/kernel.npy').astype(np.float32)
fixed_kernel = torch.tensor(fixed_kernel_np).view(
    1, 1, task_config['kernel_size'], task_config['kernel_size']
).cuda()

img_pil, downsampled_torch = generate_blurry_image('data/imgs/00287.png')
radii = torch.ones([1, 1, 1]).cuda() * np.sqrt(256 * 256 * 3)

latent = torch.nn.parameter.Parameter(
    torch.randn(
        1, config.model.in_channels,
        config.data.image_size, config.data.image_size
    ).to(device)
)
optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr_img']}])

# ─── Output dirs ───────────────────────────────────────────────────────────────
os.makedirs('results/grad_freq/fft_maps', exist_ok=True)
os.makedirs('results/grad_freq/jacobian_svd', exist_ok=True)

# ─── Silent DDIM forward (tqdm 억제) ───────────────────────────────────────────
def ddim_forward_silent(xT):
    """DDIM forward pass without tqdm output."""
    with redirect_stdout(io.StringIO()):
        return DDIM_efficient_feed_forward(xT, model, ddim_scheduler)


# ─── Frequency helpers ─────────────────────────────────────────────────────────
def _radial_mask(H, W, lo_frac, hi_frac):
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = np.sqrt(cx ** 2 + cy ** 2)
    return (r >= r_max * lo_frac) & (r < r_max * hi_frac)


def gradient_band_norms(grad_np):
    """
    grad_np: (1, C, H, W)
    Returns: (low_norm, mid_norm, high_norm, low_frac, mid_frac, high_frac)
    Low  : r < 10% r_max
    Mid  : 10%–30% r_max
    High : > 30% r_max
    """
    g = grad_np[0]
    H, W = g.shape[1], g.shape[2]
    low_m  = _radial_mask(H, W, 0.0, 0.10)
    mid_m  = _radial_mask(H, W, 0.10, 0.30)
    high_m = _radial_mask(H, W, 0.30, 1e9)

    le, me, he = 0.0, 0.0, 0.0
    for ch in g:
        pwr = np.abs(np.fft.fftshift(np.fft.fft2(ch))) ** 2
        le += pwr[low_m].sum()
        me += pwr[mid_m].sum()
        he += pwr[high_m].sum()

    total = le + me + he + 1e-12
    return np.sqrt(le), np.sqrt(me), np.sqrt(he), le/total, me/total, he/total


def save_gradient_fft_image(grad_np, iteration, save_dir):
    """latent gradient의 채널별 FFT 스펙트럼 + radial 프로파일 저장"""
    g = grad_np[0]
    H, W = g.shape[1], g.shape[2]
    cy, cx = H // 2, W // 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'∇xT L Frequency Spectrum  |  iter {iteration:04d}', fontsize=13)
    channel_names = ['Ch 0', 'Ch 1', 'Ch 2']
    radial_profiles = []

    for i, (ax, name) in enumerate(zip(axes[:3], channel_names)):
        fft_shift = np.fft.fftshift(np.fft.fft2(g[i]))
        magnitude = np.log1p(np.abs(fft_shift))
        ax.imshow(magnitude, cmap='plasma')
        ax.set_title(name)
        ax.axis('off')

        Y_idx, X_idx = np.ogrid[:H, :W]
        r = np.sqrt((X_idx - cx) ** 2 + (Y_idx - cy) ** 2).astype(int)
        counts = np.bincount(r.ravel())
        sums   = np.bincount(r.ravel(), magnitude.ravel())
        radial_profiles.append(sums / (counts + 1e-9))

    ax = axes[3]
    max_r = min(len(p) for p in radial_profiles)
    freqs = np.arange(max_r)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for profile, name, color in zip(radial_profiles, channel_names, colors):
        ax.plot(freqs, profile[:max_r], label=name, color=color, alpha=0.8)
    ax.axvspan(0,             max_r * 0.10, alpha=0.08, color='blue',  label='Low')
    ax.axvspan(max_r * 0.10, max_r * 0.30, alpha=0.08, color='green', label='Mid')
    ax.axvspan(max_r * 0.30, max_r,        alpha=0.08, color='red',   label='High')
    ax.set_xlabel('Radial frequency (px from DC)')
    ax.set_ylabel('Log magnitude (avg)')
    ax.set_title('Radial Freq Profile of ∇xT L')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/grad_fft_{iteration:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()


# ─── Jacobian SVD via finite differences + randomized SVD ─────────────────────
def compute_jacobian_svd(xT_param, n_vectors=24, eps=1e-3):
    """
    J = ∂x0/∂xT 의 randomized SVD.

    DDIM 내부가 no_grad이므로 autograd JVP 대신 finite differences 사용:
        Jv ≈ (f(xT + ε*v) - f(xT - ε*v)) / (2ε)

    Y = [Jv_1, ..., Jv_k]  (d_out × k)
    SVD(Y)의 singular values ≈ dominant singular values of J.

    Returns: singular value array (length n_vectors)
    """
    xT = xT_param.detach().clone()
    d_in = xT.numel()

    # x0 baseline (단방향 FD 절약을 위해 중앙차분 사용)
    # 각 방향에 ±ε 두 번 forward pass
    Omega = torch.randn(d_in, n_vectors, device=device)
    Omega = Omega / Omega.norm(dim=0, keepdim=True)  # 열 정규화

    Y_cols = []
    for i in range(n_vectors):
        v = Omega[:, i].reshape(xT.shape)
        xT_plus  = xT + eps * v
        xT_minus = xT - eps * v

        with torch.no_grad():
            x0_plus  = ddim_forward_silent(xT_plus).detach().cpu().float()
            x0_minus = ddim_forward_silent(xT_minus).detach().cpu().float()

        jv = (x0_plus - x0_minus) / (2 * eps)
        Y_cols.append(jv.reshape(-1))

        if (i + 1) % 4 == 0:
            print(f'  Jacobian JVP {i+1}/{n_vectors} done')

    Y = torch.stack(Y_cols, dim=1).numpy()  # (d_out, n_vectors)
    _, S, _ = np.linalg.svd(Y, full_matrices=False)
    return S  # (n_vectors,)


# ─── Main optimization loop ────────────────────────────────────────────────────
OPT_STEPS = task_config['Optimization_steps']
JACOBIAN_AT = sorted({0, OPT_STEPS // 4, OPT_STEPS // 2, OPT_STEPS - 1})
FFT_MAP_EVERY = 50   # gradient FFT 이미지 저장 주기

history = {k: [] for k in ['iter', 'low_norm', 'mid_norm', 'high_norm',
                             'low_frac', 'mid_frac', 'high_frac', 'loss']}
jacobian_svs = {}  # iter → singular values array

print(f'Optimization steps: {OPT_STEPS}')
print(f'Jacobian SVD at iterations: {JACOBIAN_AT}')

for iteration in range(OPT_STEPS):

    # ── 1. Jacobian SVD (선택된 iteration) ────────────────────────────────────
    if iteration in JACOBIAN_AT:
        print(f'\n[iter {iteration:4d}] Computing Jacobian SVD ({24} directions)...')
        sv = compute_jacobian_svd(latent, n_vectors=24, eps=1e-3)
        jacobian_svs[iteration] = sv
        np.save(f'results/grad_freq/jacobian_svd/sv_{iteration:04d}.npy', sv)
        print(f'  Top-5 SVs: {sv[:5].round(4)}')

    # ── 2. Forward + loss ──────────────────────────────────────────────────────
    optimizer.zero_grad()
    x_0_hat = DDIM_efficient_feed_forward(latent, model, ddim_scheduler)
    blurred = nn.functional.conv2d(
        x_0_hat.view(-1, 1, config.data.image_size, config.data.image_size),
        fixed_kernel, padding='same', bias=None
    ).view(1, 3, config.data.image_size, config.data.image_size)
    loss = l2_loss(blurred, downsampled_torch)
    loss.backward()

    # ── 3. Gradient frequency analysis ────────────────────────────────────────
    grad_np = latent.grad.detach().cpu().numpy()
    ln, mn, hn, lf, mf, hf = gradient_band_norms(grad_np)

    history['iter'].append(iteration)
    history['loss'].append(loss.item())
    history['low_norm'].append(float(ln))
    history['mid_norm'].append(float(mn))
    history['high_norm'].append(float(hn))
    history['low_frac'].append(float(lf))
    history['mid_frac'].append(float(mf))
    history['high_frac'].append(float(hf))

    if iteration % FFT_MAP_EVERY == 0:
        save_gradient_fft_image(grad_np, iteration, 'results/grad_freq/fft_maps')

    # ── 4. Optimizer step + sphere projection ──────────────────────────────────
    optimizer.step()
    for param in latent:
        param.data.div_(
            (param.pow(2).sum(tuple(range(param.ndim)), keepdim=True) + 1e-9).sqrt()
        )
        param.data.mul_(radii)

    if iteration % 10 == 0:
        print(f'iter {iteration:4d} | loss {loss.item():.6f} | '
              f'grad lo/mid/hi: {lf:.3f}/{mf:.3f}/{hf:.3f}')

# ─── Save gradient frequency history CSV ──────────────────────────────────────
csv_path = 'results/grad_freq/band_norms.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iter', 'loss', 'low_norm', 'mid_norm', 'high_norm',
                     'low_frac', 'mid_frac', 'high_frac'])
    for i, it in enumerate(history['iter']):
        writer.writerow([it, f"{history['loss'][i]:.6f}",
                         f"{history['low_norm'][i]:.6f}",
                         f"{history['mid_norm'][i]:.6f}",
                         f"{history['high_norm'][i]:.6f}",
                         f"{history['low_frac'][i]:.6f}",
                         f"{history['mid_frac'][i]:.6f}",
                         f"{history['high_frac'][i]:.6f}"])
print(f'Saved: {csv_path}')

# ─── Plot 1: 주파수 대역별 ℓ₂ norm 추이 ───────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
iters = history['iter']

ax = axes[0]
ax.plot(iters, history['low_norm'],  label='Low  (< 10%)', color='#3498db')
ax.plot(iters, history['mid_norm'],  label='Mid  (10–30%)', color='#2ecc71')
ax.plot(iters, history['high_norm'], label='High (> 30%)', color='#e74c3c')
ax.set_ylabel('ℓ₂ norm')
ax.set_title('Gradient ∇xT L  —  Frequency Band ℓ₂ Norms')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.stackplot(iters,
             history['low_frac'], history['mid_frac'], history['high_frac'],
             labels=['Low', 'Mid', 'High'],
             colors=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
ax.set_ylabel('Energy fraction')
ax.set_title('Gradient Energy Fraction by Band')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.semilogy(iters, history['loss'], color='#8e44ad')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss (log)')
ax.set_title('Optimization Loss')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = 'results/grad_freq/band_norms_plot.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out1}')

# ─── Plot 2: Jacobian Singular Value Spectrum ──────────────────────────────────
if jacobian_svs:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.plasma
    sorted_iters = sorted(jacobian_svs.keys())
    n_plots = len(sorted_iters)
    colors = [cmap(i / max(n_plots - 1, 1)) for i in range(n_plots)]

    # (a) log-scale singular value index plot
    ax = axes[0]
    for it, color in zip(sorted_iters, colors):
        sv = jacobian_svs[it]
        ax.semilogy(np.arange(1, len(sv) + 1), sv,
                    marker='o', markersize=5, label=f'iter {it}', color=color)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value (log scale)')
    ax.set_title('Jacobian ∂x₀/∂xT — Singular Value Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # (b) normalized singular value (decay shape)
    ax = axes[1]
    for it, color in zip(sorted_iters, colors):
        sv = jacobian_svs[it]
        sv_norm = sv / (sv[0] + 1e-12)
        ax.plot(np.arange(1, len(sv_norm) + 1), sv_norm,
                marker='o', markersize=5, label=f'iter {it}', color=color)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalized singular value (σ_i / σ_1)')
    ax.set_title('SV Spectrum (normalized) — decay → low-freq bias')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = 'results/grad_freq/jacobian_sv_spectrum.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out2}')

# ─── Plot 3: Gradient FFT radial profile 비교 (저장된 대역 norm 기반) ──────────
# 대역별 에너지의 ratio 시간 변화: 저주파 지배가 강해지는지 시각화
fig, ax = plt.subplots(figsize=(10, 5))
lo = np.array(history['low_frac'])
hi = np.array(history['high_frac'])
ratio = lo / (hi + 1e-12)
ax.plot(iters, ratio, color='#e67e22', linewidth=1.5)
ax.axhline(y=ratio[0], color='gray', linestyle='--', alpha=0.5, label=f'initial ratio={ratio[0]:.2f}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Low-freq energy / High-freq energy')
ax.set_title('Low/High Frequency Energy Ratio of ∇xT L  (↑ = stronger low-freq bias)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out3 = 'results/grad_freq/lo_hi_ratio.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out3}')

print('\nDone. Results in results/grad_freq/')
