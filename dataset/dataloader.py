import os
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
import matplotlib.pyplot as plt
from dataset.augment import ECGAugmenter, z_score_normalize, downsample_ecg

class ECGDataset(Dataset):
    def __init__(self, csv_path, data_root_dir, transform=None, oversample=False, random_seed=42):
        """
        Args:
            csv_path (str): 路径到包含 segment_id 和 label 的 CSV 文件
            data_root_dir (str): 存放 .pt 文件的根目录
            transform (callable, optional): 数据增强函数
            oversample (bool): 是否开启二分类过采样
            random_seed (int): 随机种子
        """
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.csv_path = csv_path

        df = pd.read_csv(csv_path)
        segment_ids = df['segment_id'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()

        if oversample:
            print(f"[ECGDataset] Oversampling enabled. Balancing classes...")
            segment_ids, labels = balance_binary_labels(segment_ids, labels, random_seed)
        else:
            counts = Counter(labels)
            print(f"[ECGDataset] Oversampling disabled. Class counts: {dict(counts)}")

        self.segment_ids = segment_ids
        self.labels = labels

    def __len__(self):
        if 'train' in self.csv_path:
            return len(self.segment_ids)*10
        else:
            return len(self.segment_ids)

    def __getitem__(self, idx):
        idx = idx % len(self.segment_ids)

        segment_id = self.segment_ids[idx]
        label = self.labels[idx]

        label = torch.tensor(label, dtype=torch.long)

        pt_path = os.path.join(self.data_root_dir, f"{segment_id}.pt")
        ecg = torch.load(pt_path)  # shape expected: (12, L)
        ecg = ecg.float()

        ecg = z_score_normalize(ecg) # 标准化
        ecg = downsample_ecg(ecg, orig_fs=500, target_fs=250) # 采样频率

        # 构造输入，导联缺失
        if 'train' in self.csv_path:
            ecg_input = torch.zeros_like(ecg)  # shape: [12, seq_len]
            # 随机选择 1~12 个导联
            min_leads = 1
            max_leads = 12
            num_leads_to_keep = torch.randint(min_leads, max_leads + 1, (1,)).item()
            randperm = torch.randperm(12)
            lead_indices = randperm[:num_leads_to_keep]
            last_indices = randperm[num_leads_to_keep:]
            ecg_input[lead_indices] = ecg[lead_indices]

            if self.transform is not None:
                ecg_input = self.transform(ecg_input)

            return ecg_input, label, segment_id

        ecg_input = torch.zeros_like(ecg)
        ecg_input = ecg
        # ecg_input[0] = ecg[0]  # 保留第一导联
        # ecg_input[1] = ecg[1]  # 保留第二导联
        # ecg_input[6] = ecg[6]  # 保留第七导联
        return ecg_input, label, segment_id


class ECGDataset2(Dataset):
    def __init__(self, txt_path, data_root_dir, transform=None, orig_sr=250, target_sr=250):
        """
        Args:
            txt_path (str): 路径到包含划分数据集的 TXT 文件（每行一个文件名，如 '04936'）
            data_root_dir (str): 存放 .npz 文件的根目录
            orig_sr (int): 原始采样率
            target_sr (int): 目标采样率（若相同则跳过重采样）
        """
        self.txt_path = txt_path
        self.data_root_dir = data_root_dir
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.transform = transform

        # 读取文件名列表
        with open(txt_path, 'r') as f:
            self.file_names = [line.strip() for line in f if line.strip()]
        
        if not self.file_names:
            raise ValueError(f"No file names found in {txt_path}")

        # 加载并预处理所有数据段
        self.X, self.y = self._load_and_preprocess()
        # 转换标签为数字: 'AFIB' -> 1, 'N' -> 0
        self.y = np.array([1 if label == 'AFIB' else 0 for label in self.y])

    def _load_and_preprocess(self):
        all_X = []
        all_y = []

        for fname in self.file_names:
            npz_path = os.path.join(self.data_root_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} not found, skipping.")
                continue

            try:
                data = np.load(npz_path)
                X = data['segments']  # shape: [N, L, 2]
                y = data['labels']    # shape: [N,]  # 假设 labels 是字符串数组，如 ['N', 'AFIB', ...]
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue

            if X.ndim != 3 or X.shape[-1] != 2:
                print(f"Invalid shape in {fname}: {X.shape}, skipping.")
                continue

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            raise ValueError("No valid data loaded from the provided file list.")

        X_full = np.concatenate(all_X, axis=0)  # [Total_N, L, 2]
        y_full = np.concatenate(all_y, axis=0)  # [Total_N,]

        return X_full, y_full

    def __len__(self):
        if 'train' in self.txt_path:
            return len(self.X)
        else:
            return len(self.X)

    def __getitem__(self, idx):
        idx = idx % len(self.X)

        x = self.X[idx]      # shape: [L, 2]
        y = self.y[idx]

        num_channels = x.shape[1]  # 应为 2
        L_orig = x.shape[0]

        # ===== Step 1: 重采样 =====
        if self.orig_sr != self.target_sr:
            num_samples = int(L_orig * self.target_sr / self.orig_sr)
            x_resampled = np.zeros((num_samples, num_channels), dtype=np.float32)
            for ch in range(num_channels):
                x_resampled[:, ch] = resample(x[:, ch], num_samples)
            seg = x_resampled
        else:
            seg = x.astype(np.float32)

        # ===== Step 2: 每导联标准化（per-lead z-score）=====
        seg_normalized = np.zeros_like(seg)
        for ch in range(num_channels):
            mean = np.mean(seg[:, ch])
            std = np.std(seg[:, ch])
            if std > 1e-6:
                seg_normalized[:, ch] = (seg[:, ch] - mean) / std
            else:
                seg_normalized[:, ch] = seg[:, ch] - mean

        # ===== Step 3: 扩展为 12 导联，随机放置两个导联 =====
        L_new = seg_normalized.shape[0]
        seg_12lead = np.zeros((12, L_new), dtype=np.float32)

        # 随机选择两个不同的导联位置（0~11）
        if 'train' in self.txt_path:
            lead_indices = np.random.choice(12, size=2, replace=False)
            for ch in range(num_channels):
                seg_12lead[lead_indices[ch], :] = seg_normalized[:, ch]
            seg_12lead = torch.tensor(seg_12lead, dtype=torch.float32)
            
            if self.transform is not None:
                seg_12lead = self.transform(seg_12lead)
        else:
            for ch in range(num_channels):
                seg_12lead[ch, :] = seg_normalized[:, ch]
            seg_12lead = torch.tensor(seg_12lead, dtype=torch.float32)
        
        y = torch.tensor(y, dtype=torch.long)
        return seg_12lead, y, idx
