import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

df = pd.read_csv('results/blind_deblurring_step20_metrics.csv')
metrics = ['psnr', 'ssim', 'lpips', 'loss']

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.4)

print("=" * 65)
print(f"{'Metric':<8} {'Shapiro-Wilk':>14} {'K-S':>14} {'Anderson-Darling':>18}")
print(f"{'':8} {'stat / p':>14} {'stat / p':>14} {'stat / crit(5%)':>18}")
print("=" * 65)

for i, metric in enumerate(metrics):
    data = df[metric].values

    # Shapiro-Wilk
    sw_stat, sw_p = stats.shapiro(data)

    # Kolmogorov-Smirnov (표준화 후)
    data_std = (data - data.mean()) / data.std()
    ks_stat, ks_p = stats.kstest(data_std, 'norm')

    # Anderson-Darling
    ad_result = stats.anderson(data, dist='norm')
    ad_stat = ad_result.statistic
    ad_crit_5 = ad_result.critical_values[2]  # 5% significance level

    sw_normal = "정규" if sw_p > 0.05 else "비정규"
    ks_normal = "정규" if ks_p > 0.05 else "비정규"
    ad_normal = "정규" if ad_stat < ad_crit_5 else "비정규"

    print(f"{metric:<8} {sw_stat:.4f}/{sw_p:.4f} ({sw_normal:>3})  "
          f"{ks_stat:.4f}/{ks_p:.4f} ({ks_normal:>3})  "
          f"{ad_stat:.4f}/{ad_crit_5:.4f} ({ad_normal:>3})")

    # Q-Q Plot
    ax_qq = fig.add_subplot(gs[i, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm')
    ax_qq.plot(osm, osr, 'o', color='steelblue', markersize=4, alpha=0.7)
    ax_qq.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=1.5)
    ax_qq.set_title(f'Q-Q Plot: {metric.upper()}  (R={r:.4f})', fontsize=10)
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Sample Quantiles')
    ax_qq.grid(True, alpha=0.3)

    # Histogram + 정규분포 곡선
    ax_hist = fig.add_subplot(gs[i, 1])
    ax_hist.hist(data, bins=12, density=True, color='steelblue', alpha=0.6, edgecolor='white')
    xmin, xmax = ax_hist.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    ax_hist.plot(x, stats.norm.pdf(x, data.mean(), data.std()), 'r-', linewidth=2, label='Normal fit')
    ax_hist.set_title(f'Histogram: {metric.upper()}', fontsize=10)
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Density')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # 수렴 곡선
    ax_line = fig.add_subplot(gs[i, 2])
    ax_line.plot(df['iteration'], df[metric], color='steelblue', linewidth=1.5)
    ax_line.set_title(f'Convergence: {metric.upper()}', fontsize=10)
    ax_line.set_xlabel('Iteration')
    ax_line.set_ylabel(metric.upper())
    ax_line.grid(True, alpha=0.3)

print("=" * 65)
print("\n* p > 0.05 → 정규분포 가정 기각 불가 (정규로 볼 수 있음)")
print("* Anderson-Darling: stat < crit(5%) → 정규로 볼 수 있음")

plt.suptitle('Normality Analysis of BIRD Metrics (blind_deblurring_step20)', fontsize=13, y=1.01)
plt.savefig('results/normality_analysis.png', dpi=150, bbox_inches='tight')
print("\n결과 저장: results/normality_analysis.png")
