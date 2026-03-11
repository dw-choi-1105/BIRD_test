"""
실험: Norm Projection의 문제점 분석
-------------------------------------
1. xT 방향 분포의 균일성 이탈 측정
   - 연속 iterate 간 cosine similarity (1 → 방향이 고착됨)
   - 방향 벡터의 effective dimensionality (낮아질수록 편향)
   - 현재 방향과 초기 방향 사이 각도

2. xT frequency power spectrum 직접 추적
   - Gaussian white noise는 평탄한 스펙트럼 → 편향 시 저주파 집중
   - low/mid/high band 비율 추이

3. Gaussian 분포 이탈 정량화
   - KL divergence (히스토그램 기반): KL(empirical || N(0,1))
   - Wasserstein-1 distance: scipy.stats.wasserstein_distance
   - KS statistic (기존과 비교)
"""

import os
import csv
import numpy as np
import torch
from torch import nn
import yaml
import sys
from scipy import stats
from scipy.special import kl_div

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
radii_val = float(np.sqrt(256 * 256 * 3))
radii = torch.ones([1, 1, 1]).cuda() * radii_val

latent = torch.nn.parameter.Parameter(
    torch.randn(
        1, config.model.in_channels,
        config.data.image_size, config.data.image_size
    ).to(device)
)
optimizer = torch.optim.Adam([{'params': latent, 'lr': task_config['lr_img']}])

os.makedirs('results/norm_proj', exist_ok=True)
os.makedirs('results/norm_proj/restored', exist_ok=True)

gt_np = np.array(img_pil).astype(np.uint8)

# ─── Experiment 1: Direction uniformity helpers ────────────────────────────────

def effective_dim(v_flat):
    """
    방향 벡터 좌표의 분산을 이용한 effective dimensionality.
    균일 구면 분포 → 모든 좌표 분산 동일 → effective_dim ≈ d.
    편향 분포    → 분산이 특정 좌표에 집중 → effective_dim << d.

    effective_dim = (Σ vᵢ²)² / Σ vᵢ⁴   (participation ratio 형태)
    (v를 unit vector로 정규화한 후 계산)
    """
    v = v_flat / (np.linalg.norm(v_flat) + 1e-12)
    v2 = v ** 2
    return float(v2.sum() ** 2 / (v2 ** 2).sum())


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


# ─── Experiment 2: xT frequency band powers ───────────────────────────────────

def latent_band_powers(lat_np):
    """
    lat_np: (1, C, H, W)
    Gaussian white noise → 평탄 스펙트럼 → low≈mid≈high
    Returns (low_frac, mid_frac, high_frac)
    """
    lat = lat_np[0]
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = np.sqrt(cx ** 2 + cy ** 2)

    low_m  = r < r_max * 0.10
    mid_m  = (r >= r_max * 0.10) & (r < r_max * 0.30)
    high_m = r >= r_max * 0.30

    le, me, he = 0.0, 0.0, 0.0
    for ch in lat:
        pwr = np.abs(np.fft.fftshift(np.fft.fft2(ch))) ** 2
        le += pwr[low_m].sum()
        me += pwr[mid_m].sum()
        he += pwr[high_m].sum()

    total = le + me + he + 1e-12
    return le / total, me / total, he / total


# ─── Radial PSD ───────────────────────────────────────────────────────────────

def radial_psd_profile(lat_np):
    """
    xT 각 채널의 2D FFT 파워 스펙트럼을 radial 평균하여 1D PSD 반환.
    x축: DC로부터의 주파수 거리 (픽셀 단위, 0 = DC)
    y축: 해당 주파수 반경의 평균 파워 (log scale로 보면 직관적)

    Gaussian white noise → 모든 주파수에서 동일한 파워 (평탄한 직선)
    저주파 집중 시        → 낮은 주파수(작은 x)에서 파워가 높음
    """
    lat = lat_np[0]  # (C, H, W)
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2
    max_r = int(np.sqrt(cx**2 + cy**2)) + 1

    psd_accum = np.zeros(max_r)
    psd_counts = np.zeros(max_r)

    Y_idx, X_idx = np.ogrid[:H, :W]
    r_map = np.sqrt((X_idx - cx)**2 + (Y_idx - cy)**2).astype(int)

    for ch in lat:
        ch_norm = (ch - ch.mean()) / (ch.std() + 1e-9)
        power = np.abs(np.fft.fftshift(np.fft.fft2(ch_norm)))**2
        for rr in range(max_r):
            mask = r_map == rr
            if mask.sum() > 0:
                psd_accum[rr] += power[mask].mean()
                psd_counts[rr] += 1

    psd = psd_accum / (psd_counts + 1e-12)
    freqs = np.arange(max_r)
    return freqs, psd


# ─── Experiment 4: Spatial autocorrelation ────────────────────────────────────

def spatial_autocorr_profile(lat_np, max_lag=20):
    """
    xT 각 채널의 2D 공간 자기상관 radial 프로파일 계산.
    Wiener-Khinchin: autocorr = IFFT(|FFT(x)|²)

    순수 Gaussian white noise → lag > 0 에서 autocorr ≈ 0
    공간 구조 발생 시          → lag > 0 에서 autocorr > 0 (양의 상관)

    Returns:
        lags      : 0..max_lag 배열
        profile   : 정규화된 radial autocorr (lag=0 → 1.0)
        acorr_lag1: lag=1 에서의 autocorr (핵심 지표)
        acorr_sum : lags 1..max_lag 의 절댓값 합 (전체 공간 상관량)
    """
    lat = lat_np[0]  # (C, H, W)
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2

    acorr_accum = np.zeros((H, W))
    for ch in lat:
        ch_norm = (ch - ch.mean()) / (ch.std() + 1e-9)
        power = np.abs(np.fft.fft2(ch_norm)) ** 2
        acorr_2d = np.fft.ifft2(power).real / ch_norm.size
        acorr_accum += np.fft.fftshift(acorr_2d)
    acorr_accum /= lat.shape[0]  # 채널 평균

    Y_idx, X_idx = np.ogrid[:H, :W]
    r_map = np.sqrt((X_idx - cx) ** 2 + (Y_idx - cy) ** 2).astype(int)

    profile = []
    for lag in range(max_lag + 1):
        mask = r_map == lag
        profile.append(acorr_accum[mask].mean() if mask.sum() > 0 else 0.0)

    profile = np.array(profile)
    profile_norm = profile / (profile[0] + 1e-12)  # lag=0 → 1.0

    acorr_lag1 = float(profile_norm[1])
    acorr_sum  = float(np.abs(profile_norm[1:]).sum())

    return np.arange(max_lag + 1), profile_norm, acorr_lag1, acorr_sum


# ─── Experiment 3: Gaussian divergence metrics ────────────────────────────────

def gaussian_divergences(lat_np, n_bins=200):
    """
    xT 값을 표준화 후 N(0,1)과 비교.
    Returns: ks_stat, ks_p, wasserstein1, kl_div_val
    """
    vals = lat_np.flatten().astype(np.float64)
    # 표준화
    vals_std = (vals - vals.mean()) / (vals.std() + 1e-9)

    # KS test
    ks_stat, ks_p = stats.kstest(vals_std, 'norm')

    # Wasserstein-1
    # 기준: N(0,1) 에서 len(vals) 개 샘플
    ref = np.random.randn(len(vals_std))
    w1 = stats.wasserstein_distance(vals_std, ref)

    # KL divergence (히스토그램 기반)
    lo, hi = -5.0, 5.0
    bins = np.linspace(lo, hi, n_bins + 1)
    emp_counts, _ = np.histogram(vals_std, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    ref_density = stats.norm.pdf(bin_centers)
    # 0 방지
    p = emp_counts * bin_width + 1e-10
    q = ref_density * bin_width + 1e-10
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))

    return ks_stat, ks_p, w1, kl


# ─── Snapshot plots ───────────────────────────────────────────────────────────

SNAPSHOT_ITERS = [0, 100, 300, 600, 999]
snapshots = {}  # iter → standardized flat latent values

def save_distribution_snapshot(vals_std, iteration, cos_to_init, eff_dim, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'xT Distribution  |  iter {iteration:04d}  '
                 f'cos_to_init={cos_to_init:.3f}  eff_dim={eff_dim:.1f}', fontsize=12)

    # (1) Histogram vs N(0,1)
    ax = axes[0]
    ax.hist(vals_std, bins=100, density=True, alpha=0.6, color='steelblue', label='xT (std)')
    x = np.linspace(-4, 4, 300)
    ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    ax.set_xlim(-4, 4)
    ax.set_xlabel('value')
    ax.set_ylabel('density')
    ax.set_title('Value Distribution vs N(0,1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Q-Q plot
    ax = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(vals_std, dist='norm')
    ax.scatter(osm, osr, s=0.5, alpha=0.3, color='steelblue')
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=1.5)
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.set_title(f'Q-Q Plot  (r={r:.4f})')
    ax.grid(True, alpha=0.3)

    # (3) 2D 히트맵: 공간적 분포 (flatten → 첫 채널)
    ax = axes[2]
    H = W = config.data.image_size
    ch0 = vals_std[:H * W].reshape(H, W)
    im = ax.imshow(ch0, cmap='RdBu_r', vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Spatial map (Ch 0, standardized)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/snapshot_{iteration:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()


# ─── Main optimization loop ────────────────────────────────────────────────────
OPT_STEPS = task_config['Optimization_steps']

# 초기 direction 저장
latent_init_dir = latent.detach().cpu().numpy().flatten()
latent_init_dir = latent_init_dir / (np.linalg.norm(latent_init_dir) + 1e-12)

prev_dir = latent_init_dir.copy()

history = {k: [] for k in [
    'iter', 'loss',
    # exp 1
    'cos_to_init', 'cos_to_prev', 'eff_dim',
    # exp 2
    'lat_low', 'lat_mid', 'lat_high',
    # exp 3
    'ks_stat', 'ks_p', 'wasserstein1', 'kl_div',
    # exp 4
    'acorr_lag1', 'acorr_sum',
]}

# 이론적 effective_dim (pure Gaussian → participation ratio ≈ d)
d = latent.numel()
print(f'Latent dim d = {d}')
print(f'Expected eff_dim for uniform sphere ≈ {d/3:.1f}  (participation ratio)')
print(f'Optimization steps: {OPT_STEPS}')

for iteration in range(OPT_STEPS):
    optimizer.zero_grad()
    x_0_hat = DDIM_efficient_feed_forward(latent, model, ddim_scheduler)
    blurred = nn.functional.conv2d(
        x_0_hat.view(-1, 1, config.data.image_size, config.data.image_size),
        fixed_kernel, padding='same', bias=None
    ).view(1, 3, config.data.image_size, config.data.image_size)
    loss = l2_loss(blurred, downsampled_torch)
    loss.backward()
    optimizer.step()

    # Sphere projection
    for param in latent:
        param.data.div_(
            (param.pow(2).sum(tuple(range(param.ndim)), keepdim=True) + 1e-9).sqrt()
        )
        param.data.mul_(radii)

    # ── 측정 (projection 이후) ─────────────────────────────────────────────────
    lat_np = latent.detach().cpu().numpy()   # (1, C, H, W)
    lat_flat = lat_np.flatten()

    # Exp 1: Direction
    cur_dir = lat_flat / (np.linalg.norm(lat_flat) + 1e-12)
    cos_init = cosine_sim(cur_dir, latent_init_dir)
    cos_prev = cosine_sim(cur_dir, prev_dir)
    eff_d    = effective_dim(lat_flat)
    prev_dir = cur_dir.copy()

    # Exp 2: xT frequency spectrum
    lat_low, lat_mid, lat_high = latent_band_powers(lat_np)

    # Exp 3: Gaussian divergences
    ks_stat, ks_p, w1, kl = gaussian_divergences(lat_np)

    # Exp 4: Spatial autocorrelation
    _, acorr_profile, acorr_lag1, acorr_sum = spatial_autocorr_profile(lat_np)

    history['iter'].append(iteration)
    history['loss'].append(loss.item())
    history['cos_to_init'].append(cos_init)
    history['cos_to_prev'].append(cos_prev)
    history['eff_dim'].append(eff_d)
    history['lat_low'].append(lat_low)
    history['lat_mid'].append(lat_mid)
    history['lat_high'].append(lat_high)
    history['ks_stat'].append(ks_stat)
    history['ks_p'].append(ks_p)
    history['wasserstein1'].append(w1)
    history['kl_div'].append(kl)
    history['acorr_lag1'].append(acorr_lag1)
    history['acorr_sum'].append(acorr_sum)

    # PSD + Autocorrelation 프로파일 스냅샷 저장
    if iteration in SNAPSHOT_ITERS:
        np.save(f'results/norm_proj/acorr_profile_{iteration:04d}.npy', acorr_profile)
        _, psd = radial_psd_profile(lat_np)
        np.save(f'results/norm_proj/psd_profile_{iteration:04d}.npy', psd)

    # 분포 스냅샷 저장
    if iteration in SNAPSHOT_ITERS:
        vals_std = (lat_flat - lat_flat.mean()) / (lat_flat.std() + 1e-9)
        snapshots[iteration] = vals_std.copy()
        save_distribution_snapshot(
            vals_std, iteration, cos_init, eff_d, 'results/norm_proj'
        )

    if iteration % 10 == 0:
        restored_np = process(x_0_hat, 0).astype(np.uint8)
        blurry_np   = process(downsampled_torch, 0)
        comparison  = np.concatenate([blurry_np, restored_np, gt_np], axis=1)
        Image.fromarray(comparison).save(f'results/norm_proj/restored/iter_{iteration:04d}.png')
        Image.fromarray(comparison).save('results/norm_proj/latest.png')

        print(f'iter {iteration:4d} | loss {loss.item():.5f} | '
              f'cos_init {cos_init:.4f} | eff_dim {eff_d:.1f} | '
              f'W1 {w1:.4f} | KL {kl:.4f} | '
              f'lat lo/hi {lat_low:.3f}/{lat_high:.3f} | '
              f'acorr_lag1 {acorr_lag1:.5f} | acorr_sum {acorr_sum:.4f}')

# ─── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = 'results/norm_proj/metrics.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(list(history.keys()))
    for i in range(len(history['iter'])):
        writer.writerow([f'{history[k][i]:.6f}' if isinstance(history[k][i], float)
                         else history[k][i] for k in history])
print(f'Saved: {csv_path}')

# ─── Plot 1: Direction uniformity ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
iters = history['iter']

ax = axes[0]
ax.plot(iters, history['cos_to_init'], color='#e74c3c', linewidth=1.2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Expected (uniform sphere)')
ax.set_ylabel('Cosine sim to xT(iter=0)')
ax.set_title('Direction Drift from Initial Latent')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(iters, history['cos_to_prev'], color='#3498db', linewidth=1.0, alpha=0.8)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Cosine sim to xT(iter-1)')
ax.set_title('Consecutive Direction Change  (0 → random, 1 → stuck)')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(iters, history['eff_dim'], color='#2ecc71', linewidth=1.2)
ax.axhline(d / 3, color='gray', linestyle='--', alpha=0.5, label=f'Uniform sphere ≈ {d/3:.0f}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Effective dimensionality')
ax.set_title('Participation Ratio of xT Direction  (↓ = more concentrated)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/norm_proj/direction_uniformity.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/norm_proj/direction_uniformity.png')

# ─── Plot 2: xT frequency spectrum ────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ax.plot(iters, history['lat_low'],  label='Low  (< 10%)', color='#3498db')
ax.plot(iters, history['lat_mid'],  label='Mid  (10–30%)', color='#2ecc71')
ax.plot(iters, history['lat_high'], label='High (> 30%)', color='#e74c3c')
# Gaussian white noise 기댓값: 면적 비율로 계산
H = W = config.data.image_size
cy, cx = H // 2, W // 2
Y, X = np.ogrid[:H, :W]
r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
r_max = np.sqrt(cx ** 2 + cy ** 2)
total_px = H * W
low_frac_expected  = float((r < r_max * 0.10).sum()) / total_px
mid_frac_expected  = float(((r >= r_max * 0.10) & (r < r_max * 0.30)).sum()) / total_px
high_frac_expected = float((r >= r_max * 0.30).sum()) / total_px
ax.axhline(low_frac_expected,  color='#3498db', linestyle='--', alpha=0.5,
           label=f'White noise low ≈ {low_frac_expected:.3f}')
ax.axhline(high_frac_expected, color='#e74c3c', linestyle='--', alpha=0.5,
           label=f'White noise high ≈ {high_frac_expected:.3f}')
ax.set_ylabel('Energy fraction')
ax.set_title('xT Frequency Band Power Spectrum  (dashed = expected for Gaussian white noise)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
lo = np.array(history['lat_low'])
hi = np.array(history['lat_high'])
ax.plot(iters, lo / (hi + 1e-12), color='#8e44ad', linewidth=1.2)
ax.axhline(low_frac_expected / (high_frac_expected + 1e-12),
           color='gray', linestyle='--', alpha=0.5, label='White noise baseline')
ax.set_xlabel('Iteration')
ax.set_ylabel('Low / High energy ratio')
ax.set_title('xT Low/High Frequency Ratio  (↑ from baseline = low-freq concentration)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/norm_proj/latent_freq_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/norm_proj/latent_freq_spectrum.png')

# ─── Plot 3: Gaussian divergence ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax = axes[0]
ax.semilogy(iters, history['kl_div'], color='#e74c3c', linewidth=1.2)
ax.set_ylabel('KL divergence')
ax.set_title('KL(empirical xT || N(0,1))  (↑ = farther from Gaussian)')
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ax.plot(iters, history['wasserstein1'], color='#3498db', linewidth=1.2)
ax.set_ylabel('Wasserstein-1')
ax.set_title('Wasserstein-1 distance to N(0,1)')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(iters, history['ks_stat'], color='#2ecc71', linewidth=1.2, label='KS statistic')
ax.plot(iters, history['ks_p'],    color='#f39c12', linewidth=1.0, alpha=0.7, label='KS p-value')
ax.axhline(0.05, color='gray', linestyle='--', alpha=0.5, label='p=0.05 threshold')
ax.set_xlabel('Iteration')
ax.set_ylabel('Value')
ax.set_title('KS Test vs N(0,1)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/norm_proj/gaussian_divergence.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/norm_proj/gaussian_divergence.png')

# ─── Plot 4: 스냅샷 분포 비교 (overlay) ───────────────────────────────────────
if snapshots:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.plasma
    snap_iters = sorted(snapshots.keys())
    colors = [cmap(i / max(len(snap_iters) - 1, 1)) for i in range(len(snap_iters))]
    for it, color in zip(snap_iters, colors):
        ax.hist(snapshots[it], bins=150, density=True, alpha=0.4,
                color=color, label=f'iter {it}')
    x = np.linspace(-4, 4, 300)
    ax.plot(x, stats.norm.pdf(x), 'k--', linewidth=2, label='N(0,1)')
    ax.set_xlim(-5, 5)
    ax.set_xlabel('Standardized xT value')
    ax.set_ylabel('Density')
    ax.set_title('xT Value Distribution over Iterations vs N(0,1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/norm_proj/distribution_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/norm_proj/distribution_overlay.png')

# ─── Plot 5: Spatial autocorrelation 추이 ─────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ax.plot(iters, history['acorr_lag1'], color='#e74c3c', linewidth=1.2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='White noise baseline (= 0)')
ax.set_ylabel('Autocorr at lag=1')
ax.set_title('Spatial Autocorrelation of xT at lag=1  (↑ from 0 = spatial structure emerging)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(iters, history['acorr_sum'], color='#8e44ad', linewidth=1.2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='White noise baseline (= 0)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Σ |autocorr| for lag 1..20')
ax.set_title('Total Spatial Correlation Energy  (↑ = more spatial structure than white noise)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/norm_proj/autocorr_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/norm_proj/autocorr_trend.png')

# ─── Plot 6: Autocorrelation radial 프로파일 비교 (스냅샷) ───────────────────
snap_files = sorted([
    f for f in os.listdir('results/norm_proj') if f.startswith('acorr_profile_')
])
if snap_files:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(snap_files) - 1, 1)) for i in range(len(snap_files))]
    for fname, color in zip(snap_files, colors):
        it = int(fname.split('_')[-1].split('.')[0])
        profile = np.load(f'results/norm_proj/{fname}')
        ax.plot(np.arange(len(profile)), profile, marker='o', markersize=3,
                label=f'iter {it}', color=color, alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='White noise (= 0)')
    ax.set_xlabel('Lag (pixels)')
    ax.set_ylabel('Normalized autocorrelation')
    ax.set_title('Radial Autocorrelation Profile of xT  (lag=0 → 1.0 by definition)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/norm_proj/autocorr_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/norm_proj/autocorr_profiles.png')

# ─── Plot 7: Radial PSD overlay (스냅샷 iteration 비교) ───────────────────────
psd_files = sorted([
    f for f in os.listdir('results/norm_proj') if f.startswith('psd_profile_')
])
if psd_files:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(psd_files) - 1, 1)) for i in range(len(psd_files))]

    for fname, color in zip(psd_files, colors):
        it = int(fname.split('_')[-1].split('.')[0])
        psd = np.load(f'results/norm_proj/{fname}')
        freqs = np.arange(len(psd))

        # (a) linear scale
        axes[0].plot(freqs, psd, label=f'iter {it}', color=color, alpha=0.85, linewidth=1.2)
        # (b) log-log scale (자연스러운 PSD 시각화)
        axes[1].loglog(freqs[1:], psd[1:], label=f'iter {it}', color=color, alpha=0.85, linewidth=1.2)

    for ax, scale in zip(axes, ['Linear', 'Log-Log']):
        ax.set_xlabel('Spatial frequency (pixels from DC)')
        ax.set_ylabel('Mean power')
        ax.set_title(f'Radial PSD of xT  [{scale}]\n'
                     f'(flat = white noise, left-heavy = low-freq bias)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 이론적 white noise 기준선: PSD가 상수이므로 평균값 표시
    ref_psd = np.load(f'results/norm_proj/{psd_files[0]}')
    ref_mean = ref_psd.mean()
    axes[0].axhline(ref_mean, color='gray', linestyle='--', alpha=0.5, label='White noise level')

    plt.tight_layout()
    plt.savefig('results/norm_proj/psd_radial_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/norm_proj/psd_radial_overlay.png')

print('\nDone. Results in results/norm_proj/')
