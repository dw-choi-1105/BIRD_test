import os
import yaml
import csv
import numpy as np
import tqdm
import torch
from torch import nn
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guided_diffusion.models import Model
import random
from ddim_inversion_utils import *
from utils import *
from skimage.metrics import structural_similarity as ssim
from scipy import stats
import lpips
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('configs/blind_deblurring.yml', 'r') as f:
    task_config = yaml.safe_load(f)

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


# ─────────────────────────────────────────────────────────────
# BIRD Optimization Loop (Cayley 버전)
# ─────────────────────────────────────────────────────────────

def bird_cayley(
    y: torch.Tensor,
    ddim_reverse_fn,
    degradation_model: nn.Module,
    rank: int = 64,
    lr: float = 3e-3,
    n_iter: int = 200,
    delta_t: int = 100,
    eps: float = 1e-6,
    device: str = 'cuda',
    verbose: bool = True,
) -> torch.Tensor:
    """
    BIRD + Low-rank Cayley transform 최적화.

    원본 BIRD와의 차이:
        ❌ x_T ← x_T - α∇L
        ❌ x_T ← x_T / ||x_T|| * sqrt(d)   ← norm projection (Gaussian 파괴)

        ✅ x_T = R(U,V) @ z_0               ← orthogonal reparameterization
        ✅ optimize U, V via Adam            ← Gaussian isotropy 자동 보존

    Args:
        y:                  degraded image (C, H, W)
        ddim_reverse_fn:    DDIMReverse(x_T, delta_t) → x_0
        degradation_model:  H_eta (learnable parameters)
        rank:               Cayley low-rank r (32~64 권장)
        lr:                 learning rate
        n_iter:             max iterations
        delta_t:            DDIM step size
        eps:                convergence threshold
        device:             'cuda' or 'cpu'
        verbose:            print progress

    Returns:
        x_0_hat: restored image (C, H, W)
    """
    y = y.to(device)
    degradation_model = degradation_model.to(device)

    # ── 초기 노이즈 샘플링 ──────────────────────────────────────
    z_0 = torch.randn_like(y)   # (C, H, W)

    # ── Cayley latent 모듈 초기화 ───────────────────────────────
    cayley = LowRankCayleyLatent(z_0, rank=rank).to(device)

    # ── Optimizer: U, V + degradation model 파라미터 joint 최적화 ─
    optimizer = torch.optim.Adam([
        {'params': [cayley.U, cayley.V],            'lr': lr},
        {'params': degradation_model.parameters(),  'lr': lr},
    ])

    if verbose:
        d = z_0.numel()
        print(f"[BIRD-Cayley] d={d}, rank={rank}, params={2*d*rank:,}")
        print(f"[BIRD-Cayley] Expected x_T norm: {math.sqrt(d):.2f}")
        print("-" * 60)

    # ── 최적화 루프 ─────────────────────────────────────────────
    for k in range(n_iter):
        optimizer.zero_grad()

        # x_T = R @ z_0  (norm 자동 보존, Gaussian isotropy 유지)
        x_T = cayley.get_x_T()             # (C, H, W)

        # DDIM reverse: x_T → x_0
        x_0 = ddim_reverse_fn(x_T, delta_t)   # (C, H, W)

        # Restoration loss: ||y - H_eta(x_0)||^2
        loss = torch.norm(y - degradation_model(x_0)) ** 2

        if loss.item() < eps:
            if verbose:
                print(f"[{k:4d}] Converged. Loss={loss.item():.6f}")
            break

        loss.backward()
        optimizer.step()

        if verbose and k % 20 == 0:
            norm_err = cayley.norm_error()
            print(
                f"[{k:4d}] Loss={loss.item():.4f} | "
                f"norm_error={norm_err:.6f} "
                f"(should stay ~0)"
            )

    # ── 최종 복원 이미지 ─────────────────────────────────────────
    with torch.no_grad():
        x_T_final = cayley.get_x_T()
        x_0_hat = ddim_reverse_fn(x_T_final, delta_t)

    return x_0_hat


# ─────────────────────────────────────────────────────────────
# 진단 도구: Gaussian isotropy 분석
# ─────────────────────────────────────────────────────────────

class GaussianIsotropyMonitor:
    """
    최적화 중 x_T의 Gaussian 구조 붕괴를 추적.
    논문 motivation figure 생성에 사용.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.history = {
            'iteration':        [],
            'dc_power':         [],     # FFT DC 성분 강도
            'high_freq_ratio':  [],     # 고주파 에너지 비율
            'autocorr_offdiag': [],     # 자기상관 off-diagonal 크기
            'cosine_std':       [],     # 랜덤 방향과의 cosine similarity std
            'norm_error':       [],     # norm preservation 오차
        }

    @torch.no_grad()
    def record(self, x_T: torch.Tensor, iteration: int):
        x_flat = x_T.flatten().float()
        d = x_flat.numel()
        H, W = x_T.shape[-2], x_T.shape[-1]

        self.history['iteration'].append(iteration)

        # ── 1. FFT DC 성분 (논문 Figure 2-B) ────────────────────
        x_2d = x_T[0] if x_T.dim() == 3 else x_T
        fft = torch.fft.fft2(x_2d.float())
        power = torch.abs(torch.fft.fftshift(fft)) ** 2
        total_power = power.sum().item()
        dc_power = power[H // 2, W // 2].item()
        self.history['dc_power'].append(dc_power / (total_power + 1e-8))

        # ── 2. 고/저주파 에너지 비율 (논문 Figure 1-C) ───────────
        cy, cx = H // 2, W // 2
        Y, X = torch.meshgrid(
            torch.arange(H, device=x_T.device),
            torch.arange(W, device=x_T.device),
            indexing='ij'
        )
        dist = ((Y - cy) ** 2 + (X - cx) ** 2).float().sqrt()
        max_dist = dist.max()

        low_mask  = (dist < max_dist * 0.2)
        high_mask = (dist > max_dist * 0.6)

        low_energy  = (power * low_mask ).sum().item()
        high_energy = (power * high_mask).sum().item()
        ratio = high_energy / (low_energy + 1e-8)
        self.history['high_freq_ratio'].append(ratio)

        # ── 3. Autocorrelation off-diagonal (논문 Figure 2-A 보완) ─
        # x_T가 N(0,I)이면 autocorr ≈ delta → off-diag ≈ 0
        x_norm = x_flat - x_flat.mean()
        fft_x = torch.fft.rfft(x_norm)
        autocorr = torch.fft.irfft(fft_x * fft_x.conj(), n=d)
        autocorr = autocorr / (autocorr[0].abs() + 1e-8)
        offdiag_energy = autocorr[1:min(100, d)].abs().mean().item()
        self.history['autocorr_offdiag'].append(offdiag_energy)

        # ── 4. Cosine similarity std (논문 Figure 2-A) ───────────
        x_unit = x_flat / (x_flat.norm() + 1e-8)
        n_dirs = 200
        rand_dirs = torch.randn(n_dirs, d, device=x_T.device)
        rand_dirs = rand_dirs / rand_dirs.norm(dim=1, keepdim=True)
        cos_sims = (rand_dirs @ x_unit)
        # N(0,I)이면 cos_sim std ≈ 1/sqrt(d), 편향 시 커짐
        std_normalized = cos_sims.std().item() * math.sqrt(d)
        self.history['cosine_std'].append(std_normalized)

        # ── 5. Norm error ────────────────────────────────────────
        norm_err = abs(x_flat.norm().item() - math.sqrt(d)) / math.sqrt(d)
        self.history['norm_error'].append(norm_err)

    def summary(self):
        iters = self.history['iteration']
        print("\n" + "=" * 70)
        print(f"{'Iter':>6} | {'DC%':>8} | {'HF ratio':>10} | "
              f"{'AutoCorr':>10} | {'CosStd':>8} | {'NormErr':>8}")
        print("-" * 70)
        for i in range(len(iters)):
            print(
                f"{iters[i]:6d} | "
                f"{self.history['dc_power'][i]*100:7.3f}% | "
                f"{self.history['high_freq_ratio'][i]:10.4f} | "
                f"{self.history['autocorr_offdiag'][i]:10.6f} | "
                f"{self.history['cosine_std'][i]:8.4f} | "
                f"{self.history['norm_error'][i]:8.6f}"
            )
        print("=" * 70)
        print("Note: DC% ↑, HF ratio ↓, AutoCorr ↑ → Gaussian isotropy 붕괴")
        print("      CosStd ≈ 1.0 이 이상적 (normalized by sqrt(d))")

    def plot(self, save_path: Optional[str] = None):
        """논문 motivation figure 생성."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        iters = self.history['iteration']

        axes[0, 0].plot(iters, self.history['dc_power'], 'r-o', markersize=3)
        axes[0, 0].set_title('DC Power Ratio (↑ = Gaussian 붕괴)')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('DC / Total Power')
        axes[0, 0].grid(True)

        axes[0, 1].plot(iters, self.history['high_freq_ratio'], 'b-o', markersize=3)
        axes[0, 1].set_title('High-Freq / Low-Freq Ratio (↓ = 저주파 편향)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Energy Ratio')
        axes[0, 1].grid(True)

        axes[1, 0].plot(iters, self.history['autocorr_offdiag'], 'g-o', markersize=3)
        axes[1, 0].set_title('Autocorr Off-diagonal (↑ = 독립성 위반)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Mean |autocorr[Δ≠0]|')
        axes[1, 0].grid(True)

        axes[1, 1].plot(iters, self.history['norm_error'], 'm-o', markersize=3)
        axes[1, 1].set_title('Norm Preservation Error')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('|‖x_T‖ - √d| / √d')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5,
                           label='Ideal (Cayley)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.suptitle('Gaussian Isotropy Analysis: BIRD (norm proj) vs BIRD-Cayley',
                     fontsize=13)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        plt.show()


# ─────────────────────────────────────────────────────────────
# 비교 실험: Original BIRD norm projection vs Cayley
# ─────────────────────────────────────────────────────────────

def compare_norm_projection_vs_cayley(
    y: torch.Tensor,
    ddim_reverse_fn,
    degradation_model: nn.Module,
    n_iter: int = 200,
    rank: int = 64,
    device: str = 'cuda',
) -> Tuple[dict, dict]:
    """
    Original BIRD norm projection과 Cayley 방식을 동일 조건에서 비교.
    Gaussian isotropy 붕괴 정도를 정량적으로 측정.

    Returns:
        (original_history, cayley_history): 각각 GaussianIsotropyMonitor history
    """

    # ── Original BIRD (norm projection) ─────────────────────────
    print("Running Original BIRD (norm projection)...")
    monitor_orig = GaussianIsotropyMonitor(device)

    z_0 = torch.randn_like(y).to(device)
    d = z_0.numel()
    x_T_orig = z_0.clone().requires_grad_(True)
    optimizer_orig = torch.optim.Adam([x_T_orig], lr=3e-3)

    for k in range(n_iter):
        optimizer_orig.zero_grad()
        x_0 = ddim_reverse_fn(x_T_orig, 100)
        loss = torch.norm(y - degradation_model(x_0)) ** 2
        loss.backward()
        optimizer_orig.step()

        with torch.no_grad():
            # ← 원본 BIRD norm projection
            x_T_orig.data = x_T_orig.data / x_T_orig.data.norm() * math.sqrt(d)

        if k % 20 == 0:
            monitor_orig.record(x_T_orig.detach(), k)

    # ── Cayley BIRD ───────────────────────────────────────────────
    print("\nRunning Cayley BIRD...")
    monitor_cayley = GaussianIsotropyMonitor(device)

    cayley = LowRankCayleyLatent(z_0, rank=rank).to(device)
    degradation_model_copy = type(degradation_model)().to(device)  # fresh copy
    optimizer_cayley = torch.optim.Adam([
        {'params': [cayley.U, cayley.V],                  'lr': 3e-3},
        {'params': degradation_model_copy.parameters(),   'lr': 3e-3},
    ])

    for k in range(n_iter):
        optimizer_cayley.zero_grad()
        x_T = cayley.get_x_T()
        x_0 = ddim_reverse_fn(x_T, 100)
        loss = torch.norm(y - degradation_model_copy(x_0)) ** 2
        loss.backward()
        optimizer_cayley.step()

        if k % 20 == 0:
            monitor_cayley.record(x_T.detach(), k)

    print("\n[Original BIRD - norm projection]")
    monitor_orig.summary()
    print("\n[Cayley BIRD]")
    monitor_cayley.summary()

    return monitor_orig.history, monitor_cayley.history


# ─────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────

def test_cayley_properties(d: int = 512, rank: int = 32, device: str = 'cpu'):
    """
    Cayley 변환의 핵심 성질 검증:
    1. 직교성: R^T R ≈ I
    2. Norm 보존: ||R z_0|| = ||z_0||
    3. Gradient flow 정상 여부
    """
    print(f"\n[Test] d={d}, rank={rank}")

    z_0 = torch.randn(d)
    module = LowRankCayleyLatent(z_0, rank=rank)

    # ── 1. Norm 보존 ──────────────────────────────────────────
    x_T = module.get_x_T()
    norm_in  = module.z_0.norm().item()
    norm_out = x_T.norm().item()
    print(f"  Norm in:  {norm_in:.6f}")
    print(f"  Norm out: {norm_out:.6f}")
    print(f"  Norm error: {abs(norm_in - norm_out):.2e}  (should be ~1e-5 or less)")
    assert abs(norm_in - norm_out) < 1e-3, "Norm not preserved!"

    # ── 2. Gradient flow ──────────────────────────────────────
    loss = x_T.sum()
    loss.backward()
    assert module.U.grad is not None, "Gradient not flowing to U!"
    assert module.V.grad is not None, "Gradient not flowing to V!"
    print(f"  U.grad norm: {module.U.grad.norm().item():.6f}  (should be > 0)")
    print(f"  V.grad norm: {module.V.grad.norm().item():.6f}  (should be > 0)")

    # ── 3. 초기화 근방 R ≈ I 검증 ────────────────────────────
    # init_scale=0.01이면 R@z_0 ≈ z_0
    module_small = LowRankCayleyLatent(z_0, rank=rank, init_scale=0.001)
    x_T_small = module_small.get_x_T()
    diff = (x_T_small - module_small.z_0).norm().item()
    print(f"  R≈I check (small init): ||Rz_0 - z_0|| = {diff:.6f}  (should be ~0)")

    print("  ✅ All tests passed.")



### Reproducibility
torch.set_printoptions(sci_mode=False)
ensure_reproducibility(task_config['seed'])


with open( "data/celeba_hq.yml", "r") as f:
    config1 = yaml.safe_load(f)
config = dict2namespace(config1)
model, device = load_pretrained_diffusion_model(config)

### Define the DDIM scheduler
ddim_scheduler=DDIMScheduler(beta_start=config.diffusion.beta_start, beta_end=config.diffusion.beta_end, beta_schedule=config.diffusion.beta_schedule)
ddim_scheduler.set_timesteps(config.diffusion.num_diffusion_timesteps // task_config['delta_t'])#task_config['Denoising_steps']

#scale=41
l2_loss= nn.MSELoss() #nn.L1Loss()

### Fixed kernel from kernel.npy
fixed_kernel_np = np.load('data/kernel.npy').astype(np.float32)  # (41, 41)
fixed_kernel = torch.tensor(fixed_kernel_np).view(1, 1, task_config['kernel_size'], task_config['kernel_size']).cuda()

img_pil, downsampled_torch = generate_blurry_image('data/imgs/00287.png')
radii =  torch.ones([1, 1, 1]).cuda() * (np.sqrt(256*256*3))

latent = torch.nn.parameter.Parameter(torch.randn( 1, config.model.in_channels, config.data.image_size, config.data.image_size).to(device))
optimizer = torch.optim.Adam([{'params':latent,'lr':task_config['lr_img']}])

### Metric setup
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()
gt_np = np.array(img_pil).astype(np.float32)  # HQ ground truth (0~255)

### Output directories
os.makedirs('results/blind_deblurring_fixedkernel_iter2', exist_ok=True)
os.makedirs('results/blind_deblurring_fixedkernel_freq2', exist_ok=True)
metrics_path = 'results/blind_deblurring_fixedkernel_metrics_rev.csv'
with open(metrics_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'iteration', 'psnr', 'ssim', 'lpips', 'loss',
        'latent_ks_stat', 'latent_ks_p',
        'latent_sw_stat', 'latent_sw_p',
        'freq_low_frac', 'freq_mid_frac', 'freq_high_frac',
        'img_low_frac', 'img_mid_frac', 'img_high_frac',
    ])


def save_latent_freq_image(latent_np, iteration, save_dir):
    """latent의 2D FFT 스펙트럼 및 radial 주파수 프로파일 저장"""
    lat = latent_np[0]  # (C, H, W)
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Latent Frequency Spectrum  |  iter {iteration:04d}', fontsize=13)

    channel_names = ['Ch 0', 'Ch 1', 'Ch 2']
    radial_profiles = []

    for i, (ax, name) in enumerate(zip(axes[:3], channel_names)):
        fft = np.fft.fft2(lat[i])
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))

        ax.imshow(magnitude, cmap='inferno')
        ax.set_title(name)
        ax.axis('off')

        # Radial average profile
        Y, X = np.ogrid[:H, :W]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
        counts = np.bincount(r.ravel())
        sums = np.bincount(r.ravel(), magnitude.ravel())
        radial = sums / (counts + 1e-9)
        radial_profiles.append(radial)

    # Radial frequency profile plot
    ax = axes[3]
    max_r = min(len(p) for p in radial_profiles)
    freqs = np.arange(max_r)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for profile, name, color in zip(radial_profiles, channel_names, colors):
        ax.plot(freqs[:max_r], profile[:max_r], label=name, color=color, alpha=0.8)

    # 저주파 / 중주파 / 고주파 영역 표시
    r_max = max_r
    ax.axvspan(0, r_max * 0.1, alpha=0.08, color='blue', label='Low freq')
    ax.axvspan(r_max * 0.1, r_max * 0.3, alpha=0.08, color='green', label='Mid freq')
    ax.axvspan(r_max * 0.3, r_max, alpha=0.08, color='red', label='High freq')

    ax.set_xlabel('Radial frequency (pixels from DC)')
    ax.set_ylabel('Log magnitude (avg)')
    ax.set_title('Radial Frequency Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/iter_{iteration:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()


def compute_band_powers(latent_np):
    """저/중/고주파 대역별 에너지 비율 반환 (합계 = 1.0)"""
    lat = latent_np[0]
    H, W = lat.shape[1], lat.shape[2]
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = np.sqrt(cx ** 2 + cy ** 2)

    low_mask = r < r_max * 0.1
    mid_mask = (r >= r_max * 0.1) & (r < r_max * 0.3)
    high_mask = r >= r_max * 0.3

    low_e, mid_e, high_e = 0.0, 0.0, 0.0
    for ch in lat:
        fft_shift = np.fft.fftshift(np.fft.fft2(ch))
        power = np.abs(fft_shift) ** 2
        low_e  += power[low_mask].sum()
        mid_e  += power[mid_mask].sum()
        high_e += power[high_mask].sum()

    total = low_e + mid_e + high_e + 1e-12
    return low_e / total, mid_e / total, high_e / total


for iteration in range(task_config['Optimization_steps']):
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

    if iteration % 10 == 0:
        restored_np = process(x_0_hat, 0).astype(np.float32)  # (256, 256, 3), 0~255

        # PSNR
        psnr_val = psnr_orig(gt_np, restored_np)

        # SSIM
        ssim_val = ssim(gt_np, restored_np, channel_axis=2, data_range=255.0)

        # LPIPS
        gt_tensor = torch.tensor(gt_np / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        restored_tensor = torch.tensor(restored_np / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            lpips_val = lpips_fn(gt_tensor, restored_tensor).item()

        # latent 정규성 검정
        latent_np = latent.detach().cpu().numpy()
        latent_vals = latent_np.flatten()
        latent_std = (latent_vals - latent_vals.mean()) / (latent_vals.std() + 1e-9)
        lat_ks_stat, lat_ks_p = stats.kstest(latent_std, 'norm')
        lat_sw_stat, lat_sw_p = stats.shapiro(latent_std[:5000])

        # latent 주파수 대역별 에너지 비율
        low_p, mid_p, high_p = compute_band_powers(latent_np)

        # 복원 이미지(x_0_hat) 주파수 대역별 에너지 비율
        img_np_freq = x_0_hat.detach().cpu().numpy()  # (1, 3, 256, 256)
        low_i, mid_i, high_i = compute_band_powers(img_np_freq)

        print(f'iter {iteration:4d} | loss: {loss.item():.6f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | KS_p: {lat_ks_p:.4f} | lat lo/hi: {low_p:.4f}/{high_p:.4f} | img lo/hi: {low_i:.4f}/{high_i:.4f}')

        # 중간 결과 이미지 저장
        Image.fromarray(np.concatenate([process(downsampled_torch, 0), restored_np.astype(np.uint8), gt_np.astype(np.uint8)], 1)).save(f'results/blind_deblurring_fixedkernel_iter/iter_{iteration:04d}.png')
        Image.fromarray(np.concatenate([process(downsampled_torch, 0), restored_np.astype(np.uint8), gt_np.astype(np.uint8)], 1)).save('results/blind_deblurring.png')

        # 주파수 스펙트럼 이미지 저장
        save_latent_freq_image(latent_np, iteration, 'results/blind_deblurring_fixedkernel_freq2')

        # 메트릭 CSV 저장
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                f'{psnr_val:.4f}', f'{ssim_val:.4f}', f'{lpips_val:.4f}', f'{loss.item():.6f}',
                f'{lat_ks_stat:.4f}', f'{lat_ks_p:.4f}',
                f'{lat_sw_stat:.4f}', f'{lat_sw_p:.4f}',
                f'{low_p:.6f}', f'{mid_p:.6f}', f'{high_p:.6f}',
                f'{low_i:.6f}', f'{mid_i:.6f}', f'{high_i:.6f}',
            ])
