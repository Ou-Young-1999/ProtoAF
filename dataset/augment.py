import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class ECGAugmenter:
    """
    通用ECG数据增强器，支持任意导联数。
    包含：幅度缩放、时间轴缩/放、高斯噪声、基线漂移、局部导联Mask。
    """

    def __init__(
        self,
        target_length=2500,
        p_amplitude_scale=0.5,
        scale_range=(0.7, 1.3),
        p_time_resize=0.5,
        resize_mode_prob=0.5,
        p_gaussian_noise=0.5,
        noise_std_range=(0.1, 0.6),
        p_baseline_wander=0.5,
        bw_amplitude_range=(0.1, 0.8),
        bw_freq_range=(0.1, 0.5),
        p_lead_mask=0.5,
        mask_range=(0.1,0.5),
        fs=500,
        inplace=False,
    ):
        self.target_length = target_length

        self.p_amplitude_scale = p_amplitude_scale
        self.scale_range = scale_range

        self.p_time_resize = p_time_resize
        self.resize_mode_prob = resize_mode_prob

        self.p_gaussian_noise = p_gaussian_noise
        self.noise_std_range = noise_std_range

        self.p_baseline_wander = p_baseline_wander
        self.bw_amplitude_range = bw_amplitude_range
        self.bw_freq_range = bw_freq_range

        self.p_lead_mask = p_lead_mask
        self.mask_range = mask_range

        self.fs = fs
        self.inplace = inplace

    def __call__(self, x):
        if not self.inplace:
            x = x.clone()

        if torch.rand(1).item() < self.p_amplitude_scale:
            x = self._amplitude_scale(x)

        if torch.rand(1).item() < self.p_time_resize:
            x = self._time_resize(x)

        if torch.rand(1).item() < self.p_gaussian_noise:
            x = self._add_gaussian_noise(x)

        if torch.rand(1).item() < self.p_baseline_wander:
            x = self._add_baseline_wander(x)

        if torch.rand(1).item() < self.p_lead_mask:
            x = self._lead_mask(x)

        return x

    def _amplitude_scale(self, x):
        """所有导联统一缩放因子，保持导联间空间向量关系不变"""
        low, high = self.scale_range
        scale = torch.empty(1, device=x.device).uniform_(low, high)
        x.mul_(scale)
        return x

    def _time_resize(self, x):
        """
        缩：将信号下采样至 < target_length，两侧补零对齐
        放：截取信号中间一段，上采样放大至 target_length
        输出严格为 [num_leads, target_length]
        """
        num_leads, length = x.shape
        target = self.target_length
        device = x.device

        if torch.rand(1).item() < self.resize_mode_prob:
            # === 缩模式：下采样 + 两侧补零 ===
            # 随机选择一个压缩比例 (0.5 ~ 0.9)，保证压缩后 < target
            shrink_ratio = torch.empty(1, device=device).uniform_(0.5, 0.9).item()
            new_len = max(1, int(target * shrink_ratio))

            # 使用线性插值下采样到 new_len
            x_shrunk = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=new_len, mode='linear', align_corners=False
            ).squeeze(0)

            # 居中补零至 target_length
            pad_total = target - new_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_new = torch.nn.functional.pad(x_shrunk, (pad_left, pad_right), value=0.0)

        else:
            # === 放模式：截取中间段 + 上采样放大 ===
            # 随机选择一个截取比例 (0.5 ~ 0.9)，截取长度 < target
            zoom_ratio = torch.empty(1, device=device).uniform_(0.5, 0.9).item()
            crop_len = max(1, int(target * zoom_ratio))

            # 从原始信号中心截取 crop_len 个点
            start = (length - crop_len) // 2
            start = max(0, min(start, length - crop_len))
            x_cropped = x[:, start:start + crop_len]

            # 使用线性插值上采样放大至 target_length
            x_new = torch.nn.functional.interpolate(
                x_cropped.unsqueeze(0), size=target, mode='linear', align_corners=False
            ).squeeze(0)

        x.copy_(x_new)
        return x

    def _add_gaussian_noise(self, x):
        """逐导联自适应噪声强度，基于95分位数估计信号尺度"""
        std_low, std_high = self.noise_std_range
        signal_scale = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-6
        noise_std = torch.empty(x.shape[0], 1, device=x.device).uniform_(std_low, std_high)
        noise = torch.randn_like(x) * (noise_std * signal_scale)
        x.add_(noise)
        return x

    def _add_baseline_wander(self, x):
        """每导联独立低频正弦漂移，幅度自适应信号尺度"""
        num_leads, length = x.shape
        device = x.device

        t = torch.linspace(0, length / self.fs, length, device=device)
        amp_lo, amp_hi = self.bw_amplitude_range
        freq_lo, freq_hi = self.bw_freq_range

        amplitude = torch.empty(num_leads, 1, device=device).uniform_(amp_lo, amp_hi)
        freq = torch.empty(num_leads, 1, device=device).uniform_(freq_lo, freq_hi)
        phase = torch.empty(num_leads, 1, device=device).uniform_(0, 2 * np.pi)

        wander = amplitude * torch.sin(2 * np.pi * freq * t + phase)
        signal_scale = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-6
        x.add_(wander * signal_scale)
        return x

    def _lead_mask(self, x):
        """
        随机选择一个连续时间区间，对所有导联同时Mask该区间。
        """
        num_leads, length = x.shape
        device = x.device
        lo, hi = self.mask_range

        min_mask_len = min(int(length * lo), length)
        max_mask_len = max(min_mask_len, int(length * hi))
        mask_len = torch.randint(min_mask_len, max_mask_len + 1, (1,)).item()

        # 随机选择Mask起始位置（所有导联共享同一个起始点）
        max_start = length - mask_len
        start = torch.randint(0, max_start + 1, (1,)).item()

        # 所有导联的 [start, start+mask_len) 区间同时置为mask_value
        x[:, start:start + mask_len] = 0

        return x

# ==============================
# 独立增强函数（用于单独调用）
# ==============================

def mask_interval_only(x):
    """仅做导联遮掩"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=0.0,
        p_time_resize=0.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=0.0,
        p_lead_mask=1.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)

def add_baseline_wander_only(x):
    """仅加基线漂移"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=0.0,
        p_time_resize=0.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=1.0,
        p_lead_mask=0.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)

def add_gaussian_noise_only(x):
    """仅加高斯噪声"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=0.0,
        p_time_resize=0.0,
        p_gaussian_noise=1.0,
        p_baseline_wander=0.0,
        p_lead_mask=0.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)

def add_time_resize_up_only(x):
    """仅加时间缩放"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=0.0,
        p_time_resize=1.0,
        resize_mode_prob=1.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=0.0,
        p_lead_mask=0.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)

def add_time_resize_down_only(x):
    """仅加时间缩放"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=0.0,
        p_time_resize=1.0,
        resize_mode_prob=0.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=0.0,
        p_lead_mask=0.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)

def add_amplitude_scale_up_only(x):
    """仅加幅值缩放"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=1.0,
        scale_range=(1.5, 2.0),
        p_time_resize=0.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=0.0,
        p_lead_mask=0.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)

def add_amplitude_scale_down_only(x):
    """仅加幅值缩放"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=1.0,
        scale_range=(0.1, 0.5),
        p_time_resize=0.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=0.0,
        p_lead_mask=0.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)

def apply_all_augmentations(x, **kwargs):
    """应用全部增强"""
    augmenter = ECGAugmenter(
        target_length=2500,
        p_amplitude_scale=1.0,
        p_time_resize=1.0,
        p_gaussian_noise=1.0,
        p_baseline_wander=1.0,
        p_lead_mask=1.0,
        fs=250,
        inplace=False,
    )
    return augmenter(x)


# ==============================
# Z-score 标准化（每导联）
# ==============================

def z_score_normalize(x):
    """
    对 ECG 每个导联做 Z-score 标准化（均值为0，标准差为1）
    Args:
        x (torch.Tensor): shape [12, L]
    Returns:
        torch.Tensor: normalized ECG, same shape
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


# ==============================
# 下采样到目标采样率（如 500 -> 100 Hz）
# ==============================

def downsample_ecg(x, orig_fs=500, target_fs=100):
    """
    使用简单整数下采样（要求 orig_fs % target_fs == 0）
    Args:
        x (torch.Tensor): shape [12, L]
        orig_fs (int): 原始采样率
        target_fs (int): 目标采样率
    Returns:
        torch.Tensor: downsampled ECG, shape [12, L_new]
    """
    if orig_fs % target_fs != 0:
        raise ValueError(f"Only integer downsampling supported. {orig_fs} not divisible by {target_fs}")

    factor = orig_fs // target_fs  # 500 -> 100 => factor = 5
    # 取每隔 factor 个点（简单下采样，实际可加抗混叠滤波）
    return x[:, ::factor]


# ==============================
# 可视化函数
# ==============================

def plot_ecg_comparison(original, augmented, title, save_path, fs=250, leads_to_show=12):
    """
    绘制原始 vs 增强后的 ECG（前 N 个导联）
    """
    L = original.shape[1]
    time_sec = torch.linspace(0, L / fs, L).numpy()

    fig, axes = plt.subplots(leads_to_show, 1, figsize=(12, 2 * leads_to_show), sharex=True)
    if leads_to_show == 1:
        axes = [axes]

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for i in range(leads_to_show):
        axes[i].plot(time_sec, original[i].numpy(), color='black', linewidth=2, label='Original')
        axes[i].plot(time_sec, augmented[i].numpy(), color='red', linewidth=1.5, alpha=0.5, label='Augmented')
        axes[i].set_ylabel(lead_names[i], fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==============================
# 主程序：测试与可视化
# ==============================

if __name__ == "__main__":

    pt_path = "/home/oyxq/PythonProjects/dataset/CPSC2018/PTDB/seg_000015.pt"
    output_dir = "./augmentation_results"
    fs = 500

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载 ECG
    ecg_tensor = torch.load(pt_path)  # shape [12, L]
    if ecg_tensor.ndim != 2 or ecg_tensor.shape[0] != 12:
        raise ValueError(f"Expected [12, L] tensor, got {ecg_tensor.shape}")

    print(f"Loaded ECG with shape: {ecg_tensor.shape}, duration: {ecg_tensor.shape[1] / fs:.1f} seconds")

    ecg_tensor = z_score_normalize(ecg_tensor) # 标准化
    ecg_tensor = downsample_ecg(ecg_tensor, orig_fs=500, target_fs=250) # 采样频率
    # ecg_tensor = ecg_tensor[0:2,:]

    # 1. 仅遮掩
    masked = mask_interval_only(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, masked,
        title="ECG Augmentation: Lead Interval Masking Only",
        save_path=os.path.join(output_dir, "maskedinterval.png"),
    )

    # 2. 仅高斯噪声
    noised = add_gaussian_noise_only(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, noised,
        title="ECG Augmentation: Gaussian Noise Only",
        save_path=os.path.join(output_dir, "noised.png"),
    )

    # 3. 仅基线漂移
    wandered = add_baseline_wander_only(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, wandered,
        title="ECG Augmentation: Baseline Wander Only",
        save_path=os.path.join(output_dir, "wandered.png"),
    )

    #
    time = add_time_resize_up_only(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, time,
        title="ECG Augmentation: Time Resize Only",
        save_path=os.path.join(output_dir, "time_up.png"),
    )
    time = add_time_resize_down_only(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, time,
        title="ECG Augmentation: Time Resize Only",
        save_path=os.path.join(output_dir, "time_down.png"),
    )

    #
    amp = add_amplitude_scale_up_only(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, amp,
        title="ECG Augmentation: Amplitude Scale Only",
        save_path=os.path.join(output_dir, "amp_up.png"),
    )
    amp = add_amplitude_scale_down_only(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, amp,
        title="ECG Augmentation: Amplitude Scale Only",
        save_path=os.path.join(output_dir, "amp_down.png"),
    )

    # 4. 全部增强
    all_aug = apply_all_augmentations(ecg_tensor.clone())
    plot_ecg_comparison(
        ecg_tensor, all_aug,
        title="ECG Augmentation: All",
        save_path=os.path.join(output_dir, "all_aug.png"),
        fs=fs
    )

    # # 5. Z-score 标准化
    # zscored = z_score_normalize(ecg_tensor.clone())
    # plot_ecg_comparison(
    #     ecg_tensor, zscored,
    #     title="ECG Preprocessing: Z-Score Normalization (per lead)",
    #     save_path=os.path.join(output_dir, "zscore.png"),
    #     fs=fs
    # )

    # # 6. 下采样到 100 Hz
    # downsampled = downsample_ecg(ecg_tensor.clone(), orig_fs=fs, target_fs=100)
    # print(f"Downsampled from {ecg_tensor.shape[1]} to {downsampled.shape[1]} samples (500Hz → 100Hz)")

    # # 可视化下采样（需调整时间轴）
    # L_orig = ecg_tensor.shape[1]
    # L_new = downsampled.shape[1]
    # time_orig = torch.linspace(0, L_orig / fs, L_orig).numpy()
    # time_new = torch.linspace(0, L_new / 100, L_new).numpy()

    # fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)  # 只画前3导联避免太密
    # lead_names = ['I', 'II', 'V1']
    # for i in range(3):
    #     axes[i].plot(time_orig, ecg_tensor[i].numpy(), color='black', linewidth=1.0, label='Original (500Hz)')
    #     axes[i].scatter(time_new, downsampled[i].numpy(), color='red', s=8, alpha=0.7, label='Downsampled (100Hz)')
    #     axes[i].set_ylabel(lead_names[i])
    #     axes[i].grid(True, linestyle='--', alpha=0.5)
    #     if i == 0:
    #         axes[i].legend()
    # axes[-1].set_xlabel('Time (s)')
    # plt.suptitle("ECG Downsampling: 500 Hz → 100 Hz")
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(os.path.join(output_dir, "downsampled.png"), dpi=150)
    # plt.close()

    print(f"✅ All augmentation visualizations saved to: {output_dir}")