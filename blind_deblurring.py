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
os.makedirs('results/blind_deblurring_fixedkernel_iter', exist_ok=True)
os.makedirs('results/blind_deblurring_fixedkernel_freq', exist_ok=True)
metrics_path = 'results/blind_deblurring_fixedkernel_metrics.csv'
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
    for param in latent:
        param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
        param.data.mul_(radii)

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
        save_latent_freq_image(latent_np, iteration, 'results/blind_deblurring_fixedkernel_freq')

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
