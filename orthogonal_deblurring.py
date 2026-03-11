"""
Orthogonal Manifold Deblurring via Cayley Reparameterization
=============================================================
기존 norm projection (sphere projection) 방식 대신,
xT = Q · z0  (Q ∈ O(d), z0 fixed reference noise)
로 reparameterize하여 최적화.

Cayley map: Q = (I + A)^{-1} (I - A),  A = U·V^T - V·U^T (skew-symmetric, rank-2r)
Woodbury identity로 O(d·r) 효율 구현 — d×d 행렬 절대 미생성.

Comparison metrics (norm_projection_analysis.py와 동일):
  - Direction cosine drift (cos_to_init, cos_to_prev)
  - Effective dimensionality (participation ratio)
  - Frequency band powers (low/mid/high)
  - Gaussian divergences (KS, Wasserstein-1, KL)
  - Spatial autocorrelation (acorr_lag1, acorr_sum)
  - Radial PSD snapshots
  - Reconstruction quality (PSNR, SSIM, LPIPS)
"""

import os
import csv
import yaml
import sys
import math
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from skimage.metrics import structural_similarity as ssim_fn
import lpips

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guided_diffusion.models import Model
from ddim_inversion_utils import *
from utils import *

# ─── Config ───────────────────────────────────────────────────────────────────

with open('configs/blind_deblurring.yml', 'r') as f:
    task_config = yaml.safe_load(f)

with open('data/celeba_hq.yml', 'r') as f:
    config1 = yaml.safe_load(f)
config = dict2namespace(config1)

OPT_STEPS = task_config['Optimization_steps']
RANK = 64          # low-rank Cayley rank r (default 64)
SNAPSHOT_ITERS = [0, 100, 300, 600, 999]
LOG_EVERY = 10

# ─── Model & data ─────────────────────────────────────────────────────────────

torch.set_printoptions(sci_mode=False)
ensure_reproducibility(task_config['seed'])

model, device = load_pretrained_diffusion_model(config)
ddim_scheduler = DDIMScheduler(
    beta_start=config.diffusion.beta_start,
    beta_end=config.diffusion.beta_end,
    beta_schedule=config.diffusion.beta_schedule,
)
ddim_scheduler.set_timesteps(config.diffusion.num_diffusion_timesteps // task_config['delta_t'])

l2_loss = nn.MSELoss()

fixed_kernel_np = np.load('data/kernel.npy').astype(np.float32)
fixed_kernel = torch.tensor(fixed_kernel_np).view(
    1, 1, task_config['kernel_size'], task_config['kernel_size']
).to(device)

img_pil, downsampled_torch = generate_blurry_image('data/imgs/00287.png')
gt_np = np.array(img_pil).astype(np.uint8)

# ─── LowRankCayleyLatent ──────────────────────────────────────────────────────

class LowRankCayleyLatent(nn.Module):
    """
    x_T = R(U,V) @ z_0 를 학습 가능한 파라미터 U, V로 표현.

    Args:
        z_0:   초기 노이즈 텐서 (임의 shape). 내부적으로 flatten해서 관리.
        rank:  Low-rank 파라미터 r. 총 2dr 파라미터. (권장: 32~64)
        init_scale: U, V 초기화 스케일. 작을수록 R ≈ I (x_T ≈ z_0 에서 시작).
    """

    def __init__(
        self,
        z_0: torch.Tensor,
        rank: int = 64,
        init_scale: float = 0.01,
    ):
        super().__init__()

        self.rank = rank
        self.original_shape = z_0.shape
        self.d = z_0.numel()

        # z_0 고정: norm = sqrt(d) 로 정규화
        z_flat = z_0.detach().flatten().float()
        z_flat = z_flat / z_flat.norm() * math.sqrt(self.d)
        self.register_buffer('z_0', z_flat)

        # 학습 파라미터: U, V ∈ R^{d × r}
        # init_scale 작게 → R ≈ I → 최적화 초기에 x_T ≈ z_0
        self.U = nn.Parameter(torch.randn(self.d, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(self.d, rank) * init_scale)

    # ── Woodbury 기반 Cayley matrix-vector product ──────────────

    def _cayley_mv(self, z: torch.Tensor) -> torch.Tensor:
        """
        R @ z 를 효율적으로 계산.

        R = (I + A/2)^{-1}(I - A/2)
        A/2 = P S P^T,  P = [U|V],  S = [[0,-I/2],[I/2,0]]

        Step 1: w = (I - A/2) @ z = z - P S (P^T z)
        Step 2: R @ z = (I + A/2)^{-1} @ w
                      = w - P C^{-1} (P^T w)
            C = S^{-1} + P^T P  (2r×2r, cheap to invert)
        """
        U, V = self.U, self.V
        r = self.rank

        # ── Step 1: w = (I - A/2) @ z ─────────────────────────
        # P^T z = [U^T z; V^T z]
        UTz = U.T @ z   # (r,)
        VTz = V.T @ z   # (r,)

        # S @ [UTz; VTz] = [-VTz/2; UTz/2]
        # P @ (S @ P^T z) = U@(-VTz/2) + V@(UTz/2)
        w = z - (U @ (-VTz / 2) + V @ (UTz / 2))   # (d,)

        # ── Step 2: (I + A/2)^{-1} @ w ────────────────────────
        # C = S^{-1} + P^T P
        # S^{-1} = [[0, 2I], [-2I, 0]]
        UTU = U.T @ U   # (r, r)
        UTV = U.T @ V   # (r, r)
        VTU = V.T @ U   # (r, r)
        VTV = V.T @ V   # (r, r)

        two_I = 2.0 * torch.eye(r, device=U.device, dtype=U.dtype)

        # C = [[U^TU,       2I + U^TV ],
        #      [-2I + V^TU, V^TV      ]]
        C = torch.cat([
            torch.cat([UTU,             two_I + UTV], dim=1),
            torch.cat([-two_I + VTU,    VTV        ], dim=1),
        ], dim=0)   # (2r, 2r)

        # P^T w = [U^T w; V^T w]
        PTw = torch.cat([U.T @ w, V.T @ w], dim=0)   # (2r,)

        # C^{-1} @ P^T w
        C_inv_PTw = torch.linalg.solve(C, PTw)        # (2r,)

        # P @ (C^{-1} @ P^T w)
        P_Cinv_PTw = U @ C_inv_PTw[:r] + V @ C_inv_PTw[r:]  # (d,)

        return w - P_Cinv_PTw   # (d,)

    # ── Public interface ────────────────────────────────────────

    def get_x_T(self) -> torch.Tensor:
        """x_T = R @ z_0, original shape으로 반환."""
        x_T_flat = self._cayley_mv(self.z_0)
        return x_T_flat.reshape(self.original_shape)

    @torch.no_grad()
    def norm_error(self) -> float:
        """
        Norm preservation 오차 확인용.
        이상적으로는 0에 가까워야 함.
        ||x_T|| - sqrt(d) ≈ 0 이면 Gaussian isotropy 유지 중.
        """
        norm = self.get_x_T().norm().item()
        expected = math.sqrt(self.d)
        return abs(norm - expected) / expected  # relative error


# ─── Cayley reparameterization ────────────────────────────────────────────────

D = config.model.in_channels * config.data.image_size * config.data.image_size  # 196608
# z0_raw = torch.randn(1, config.model.in_channels,
#                      config.data.image_size, config.data.image_size,
#                      device=device)
# cayley_module = LowRankCayleyLatent(z0_raw, rank=RANK).to(device)

latent = torch.nn.parameter.Parameter(torch.randn( 1, config.model.in_channels, config.data.image_size, config.data.image_size).to(device))
optimizer = torch.optim.Adam([{'params':latent,'lr':task_config['lr_img']}])

# ─── LPIPS ────────────────────────────────────────────────────────────────────

lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()

# ─── Output dirs ──────────────────────────────────────────────────────────────

OUT = 'results/orth_deblur'
os.makedirs(f'{OUT}/restored', exist_ok=True)


# ─── Analysis helpers (identical to norm_projection_analysis.py) ──────────────

def effective_dim(v_flat: np.ndarray) -> float:
    v = v_flat / (np.linalg.norm(v_flat) + 1e-12)
    v2 = v ** 2
    return float(v2.sum() ** 2 / ((v2 ** 2).sum() + 1e-30))


def latent_band_powers(lat_np: np.ndarray):
    lat = lat_np[0]
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = np.sqrt(cx ** 2 + cy ** 2)
    low_m  = r <  r_max * 0.1
    mid_m  = (r >= r_max * 0.1) & (r < r_max * 0.3)
    high_m = r >= r_max * 0.3
    le = me = he = 0.0
    for ch in lat:
        pwr = np.abs(np.fft.fftshift(np.fft.fft2(ch))) ** 2
        le += pwr[low_m].sum()
        me += pwr[mid_m].sum()
        he += pwr[high_m].sum()
    total = le + me + he + 1e-12
    return le / total, me / total, he / total


def radial_psd_profile(lat_np: np.ndarray):
    lat = lat_np[0]
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2
    max_r = int(np.sqrt(cx ** 2 + cy ** 2)) + 1
    psd_accum  = np.zeros(max_r)
    psd_counts = np.zeros(max_r)
    Y_idx, X_idx = np.ogrid[:H, :W]
    r_map = np.sqrt((X_idx - cx) ** 2 + (Y_idx - cy) ** 2).astype(int)
    for ch in lat:
        ch_norm = (ch - ch.mean()) / (ch.std() + 1e-9)
        power = np.abs(np.fft.fftshift(np.fft.fft2(ch_norm))) ** 2
        for rr in range(max_r):
            mask = r_map == rr
            if mask.sum() > 0:
                psd_accum[rr]  += power[mask].mean()
                psd_counts[rr] += 1
    return np.arange(max_r), psd_accum / (psd_counts + 1e-12)


def spatial_autocorr_profile(lat_np: np.ndarray, max_lag: int = 20):
    lat = lat_np[0]
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2
    acorr_accum = np.zeros((H, W))
    for ch in lat:
        ch_norm = (ch - ch.mean()) / (ch.std() + 1e-9)
        power = np.abs(np.fft.fft2(ch_norm)) ** 2
        acorr_2d = np.fft.ifft2(power).real / ch_norm.size
        acorr_accum += np.fft.fftshift(acorr_2d)
    acorr_accum /= lat.shape[0]
    Y_idx, X_idx = np.ogrid[:H, :W]
    r_map = np.sqrt((X_idx - cx) ** 2 + (Y_idx - cy) ** 2).astype(int)
    profile = []
    for lag in range(max_lag + 1):
        mask = r_map == lag
        profile.append(acorr_accum[mask].mean() if mask.sum() > 0 else 0.0)
    profile = np.array(profile)
    profile_norm = profile / (profile[0] + 1e-12)
    return (np.arange(max_lag + 1), profile_norm,
            float(profile_norm[1]), float(np.abs(profile_norm[1:]).sum()))


def gaussian_divergences(lat_np: np.ndarray, n_bins: int = 200):
    vals = lat_np.flatten().astype(np.float64)
    std_vals = (vals - vals.mean()) / (vals.std() + 1e-9)
    ks_stat, ks_p = stats.kstest(std_vals, 'norm')
    w1 = stats.wasserstein_distance(std_vals, np.random.standard_normal(len(std_vals)))
    hist, bin_edges = np.histogram(std_vals, bins=n_bins, density=True)
    bin_w = bin_edges[1] - bin_edges[0]
    ref_pdf = stats.norm.pdf((bin_edges[:-1] + bin_edges[1:]) / 2)
    p = hist * bin_w + 1e-12
    q = ref_pdf * bin_w + 1e-12
    kl = float(np.sum(p * np.log(p / q)))
    return float(ks_stat), float(ks_p), float(w1), kl


def save_distribution_snapshot(vals_std, iteration, cos_init, eff_d, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.linspace(-4, 4, 300)
    axes[0].hist(vals_std, bins=100, density=True, alpha=0.6,
                 color='steelblue', label=f'iter {iteration}')
    axes[0].plot(x, stats.norm.pdf(x), 'r--', label='N(0,1)')
    axes[0].set_title(f'xT distribution  iter={iteration}  cos_init={cos_init:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    qq = stats.probplot(vals_std[:5000], dist='norm')
    axes[1].scatter(qq[0][0], qq[0][1], s=1, alpha=0.4)
    axes[1].plot(qq[0][0], qq[1][0] * qq[0][0] + qq[1][1], 'r-', linewidth=1.5)
    axes[1].set_title(f'Q-Q plot  eff_dim={eff_d:.1f}')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dist_snapshot_{iteration:04d}.png', dpi=120, bbox_inches='tight')
    plt.close()


# ─── Metric history ───────────────────────────────────────────────────────────

history = {k: [] for k in [
    'iter', 'loss',
    'cos_to_init', 'cos_to_prev', 'eff_dim',
    'lat_low', 'lat_mid', 'lat_high',
    'ks_stat', 'ks_p', 'wasserstein1', 'kl_div',
    'acorr_lag1', 'acorr_sum',
    'psnr', 'ssim', 'lpips',
]}
snapshots = {}

csv_path = f'{OUT}/metrics.csv'
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(list(history.keys()))

print(f'Latent dim d = {D}')
print(f'Cayley rank r = {RANK}  (2r = {2*RANK} free directions per step)')
print(f'Optimization steps: {OPT_STEPS}')

# ─── Optimization loop ────────────────────────────────────────────────────────

prev_dir = None

for iteration in range(OPT_STEPS):
    optimizer.zero_grad()
    x_0_hat = DDIM_efficient_feed_forward(latent, model, ddim_scheduler)

    blurred_xt = nn.functional.conv2d(x_0_hat.view(-1, 1, config.data.image_size, config.data.image_size), fixed_kernel, padding="same", bias=None).view(1, 3, config.data.image_size, config.data.image_size)
    loss = l2_loss(blurred_xt, downsampled_torch)
    loss.backward()
    optimizer.step()

    #Project to the Sphere of radius sqrt(D)
    #for param in latent:
    #    param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
    #    param.data.mul_(radii)
    for param in latent:
        cayley = LowRankCayleyLatent(param, rank=64).to(device)
        param = cayley.get_x_T()

    # ── Analysis (every iteration for CSV) ────────────────────────────────────
    lat_np   = latent.detach().cpu().numpy()
    lat_flat = lat_np.flatten()

    # Exp 1: direction drift
    cur_dir  = lat_flat / (np.linalg.norm(lat_flat) + 1e-12)
    cos_init = float(np.dot(cur_dir, snapshots.get('init_dir', cur_dir)))
    if 'init_dir' not in snapshots:
        snapshots['init_dir'] = cur_dir.copy()
        cos_init = 1.0
    cos_prev = float(np.dot(cur_dir, prev_dir)) if prev_dir is not None else 1.0
    eff_d    = effective_dim(lat_flat)
    prev_dir = cur_dir.copy()

    # Exp 2: band powers
    lat_low, lat_mid, lat_high = latent_band_powers(lat_np)

    # Exp 3: Gaussian divergences
    ks_stat, ks_p, w1, kl = gaussian_divergences(lat_np)

    # Exp 4: spatial autocorrelation
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

    # PSD + autocorr profile snapshots
    if iteration in SNAPSHOT_ITERS:
        np.save(f'{OUT}/acorr_profile_{iteration:04d}.npy', acorr_profile)
        _, psd = radial_psd_profile(lat_np)
        np.save(f'{OUT}/psd_profile_{iteration:04d}.npy', psd)

    # Distribution snapshots
    if iteration in SNAPSHOT_ITERS:
        vals_std = (lat_flat - lat_flat.mean()) / (lat_flat.std() + 1e-9)
        snapshots[iteration] = vals_std.copy()
        save_distribution_snapshot(vals_std, iteration, cos_init, eff_d, OUT)

    # Reconstruction quality + image saving every LOG_EVERY iters
    if iteration % LOG_EVERY == 0:
        restored_np = process(x_0_hat, 0).astype(np.float32)
        blurry_np   = process(downsampled_torch, 0)

        psnr_val = psnr_orig(gt_np.astype(np.float32), restored_np)
        ssim_val = ssim_fn(gt_np.astype(np.float32), restored_np,
                           channel_axis=2, data_range=255.0)
        gt_t  = torch.tensor(gt_np / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        re_t  = torch.tensor(restored_np / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            lpips_val = lpips_fn(gt_t, re_t).item()

        history['psnr'].append(psnr_val)
        history['ssim'].append(ssim_val)
        history['lpips'].append(lpips_val)

        comparison = np.concatenate(
            [blurry_np, restored_np.astype(np.uint8), gt_np], axis=1
        )
        Image.fromarray(comparison).save(f'{OUT}/restored/iter_{iteration:04d}.png')
        Image.fromarray(comparison).save(f'{OUT}/latest.png')

        print(f'iter {iteration:4d} | loss {loss.item():.5f} | '
              f'PSNR {psnr_val:.2f} | SSIM {ssim_val:.4f} | '
              f'cos_init {cos_init:.4f} | eff_dim {eff_d:.1f} | '
              f'W1 {w1:.4f} | KL {kl:.4f} | '
              f'lat lo/hi {lat_low:.3f}/{lat_high:.3f} | '
              f'acorr_lag1 {acorr_lag1:.5f}')
    else:
        history['psnr'].append(float('nan'))
        history['ssim'].append(float('nan'))
        history['lpips'].append(float('nan'))

    with open(csv_path, 'a', newline='') as f:
        i = len(history['iter']) - 1
        csv.writer(f).writerow([
            f'{history[k][i]:.6f}' if isinstance(history[k][i], float) and not np.isnan(history[k][i])
            else ('' if isinstance(history[k][i], float) and np.isnan(history[k][i]) else history[k][i])
            for k in history
        ])


# ─── Plots ────────────────────────────────────────────────────────────────────

iters = history['iter']

# Plot 1: Direction uniformity
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axes[0].plot(iters, history['cos_to_init'], color='#2980b9', linewidth=1.2)
axes[0].set_ylabel('Cosine to init')
axes[0].set_title('xT Direction Drift  (cos=1 → same direction as init)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, history['cos_to_prev'], color='#27ae60', linewidth=1.2)
axes[1].set_ylabel('Cosine to prev')
axes[1].set_title('Step-to-Step Cosine Similarity  (1 → no angular change)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(iters, history['eff_dim'], color='#8e44ad', linewidth=1.2)
axes[2].axhline(D / 3, color='gray', linestyle='--', alpha=0.6,
                label=f'Uniform sphere ≈ {D/3:.0f}')
axes[2].set_ylabel('Effective dim (participation ratio)')
axes[2].set_title('Direction Uniformity  (↓ = more anisotropic)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlabel('Iteration')
plt.tight_layout()
plt.savefig(f'{OUT}/direction_uniformity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT}/direction_uniformity.png')

# Plot 2: Frequency band powers
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, key, color, label in zip(
    axes,
    ['lat_low', 'lat_mid', 'lat_high'],
    ['#3498db', '#2ecc71', '#e74c3c'],
    ['Low (<10% r_max)', 'Mid (10–30%)', 'High (>30%)'],
):
    ax.plot(iters, history[key], color=color, linewidth=1.2, label=label)
    ax.set_ylabel('Power fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
axes[0].set_title('xT Frequency Band Powers (Cayley optimization)')
axes[-1].set_xlabel('Iteration')
plt.tight_layout()
plt.savefig(f'{OUT}/latent_freq_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT}/latent_freq_spectrum.png')

# Plot 3: Gaussian divergence
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, key, color, title in zip(
    axes.flatten(),
    ['ks_stat', 'ks_p', 'wasserstein1', 'kl_div'],
    ['#e67e22', '#16a085', '#8e44ad', '#c0392b'],
    ['KS Statistic', 'KS p-value', 'Wasserstein-1', 'KL Divergence'],
):
    ax.plot(iters, history[key], color=color, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)
axes[0][1].axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
axes[0][1].legend()
plt.suptitle('xT Gaussian Divergence Metrics (Cayley)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/gaussian_divergence.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT}/gaussian_divergence.png')

# Plot 4: Distribution overlay
if snapshots:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_ref = np.linspace(-5, 5, 400)
    ax.plot(x_ref, stats.norm.pdf(x_ref), 'k--', linewidth=2, label='N(0,1)', zorder=10)
    cmap = plt.cm.plasma
    snap_iters = [k for k in SNAPSHOT_ITERS if isinstance(k, int) and k in snapshots]
    colors = [cmap(i / max(len(snap_iters) - 1, 1)) for i in range(len(snap_iters))]
    for it, color in zip(snap_iters, colors):
        if it in snapshots:
            ax.hist(snapshots[it], bins=100, density=True, alpha=0.35,
                    color=color, label=f'iter {it}')
    ax.set_title('xT Value Distribution over Iterations vs N(0,1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/distribution_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT}/distribution_overlay.png')

# Plot 5: Spatial autocorrelation trend
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(iters, history['acorr_lag1'], color='#e74c3c', linewidth=1.2)
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5, label='White noise baseline')
axes[0].set_ylabel('Autocorr at lag=1')
axes[0].set_title('Spatial Autocorrelation of xT at lag=1 (Cayley)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, history['acorr_sum'], color='#8e44ad', linewidth=1.2)
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5, label='White noise baseline')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Σ |autocorr| lag 1..20')
axes[1].set_title('Total Spatial Correlation Energy (Cayley)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/autocorr_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT}/autocorr_trend.png')

# Plot 6: Autocorrelation radial profile overlay
snap_files = sorted([f for f in os.listdir(OUT) if f.startswith('acorr_profile_')])
if snap_files:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(snap_files) - 1, 1)) for i in range(len(snap_files))]
    for fname, color in zip(snap_files, colors):
        it = int(fname.split('_')[-1].split('.')[0])
        profile = np.load(f'{OUT}/{fname}')
        ax.plot(np.arange(len(profile)), profile, marker='o', markersize=3,
                label=f'iter {it}', color=color, alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='White noise')
    ax.set_xlabel('Lag (pixels)')
    ax.set_ylabel('Normalized autocorrelation')
    ax.set_title('Radial Autocorrelation Profile of xT (Cayley)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/autocorr_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT}/autocorr_profiles.png')

# Plot 7: Radial PSD overlay
psd_files = sorted([f for f in os.listdir(OUT) if f.startswith('psd_profile_')])
if psd_files:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(psd_files) - 1, 1)) for i in range(len(psd_files))]
    for fname, color in zip(psd_files, colors):
        it = int(fname.split('_')[-1].split('.')[0])
        psd = np.load(f'{OUT}/{fname}')
        freqs = np.arange(len(psd))
        axes[0].plot(freqs, psd, label=f'iter {it}', color=color, alpha=0.85, linewidth=1.2)
        axes[1].loglog(freqs[1:], psd[1:], label=f'iter {it}', color=color, alpha=0.85, linewidth=1.2)
    for ax, scale in zip(axes, ['Linear', 'Log-Log']):
        ax.set_xlabel('Spatial frequency (pixels from DC)')
        ax.set_ylabel('Mean power')
        ax.set_title(f'Radial PSD of xT (Cayley) [{scale}]')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ref_psd = np.load(f'{OUT}/{psd_files[0]}')
    axes[0].axhline(ref_psd.mean(), color='gray', linestyle='--', alpha=0.5, label='White noise level')
    plt.tight_layout()
    plt.savefig(f'{OUT}/psd_radial_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT}/psd_radial_overlay.png')

# Plot 8: Reconstruction quality
valid_iters = [history['iter'][i] for i in range(len(iters))
               if not np.isnan(history['psnr'][i])]
psnr_vals  = [v for v in history['psnr']  if not np.isnan(v)]
ssim_vals  = [v for v in history['ssim']  if not np.isnan(v)]
lpips_vals = [v for v in history['lpips'] if not np.isnan(v)]

if valid_iters:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(valid_iters, psnr_vals,  color='#2980b9', linewidth=1.5)
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Reconstruction Quality — Cayley Orthogonal Optimization')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(valid_iters, ssim_vals,  color='#27ae60', linewidth=1.5)
    axes[1].set_ylabel('SSIM')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(valid_iters, lpips_vals, color='#e74c3c', linewidth=1.5)
    axes[2].set_ylabel('LPIPS (↓ better)')
    axes[2].set_xlabel('Iteration')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/reconstruction_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT}/reconstruction_quality.png')

print(f'\nDone. All results in {OUT}/')
