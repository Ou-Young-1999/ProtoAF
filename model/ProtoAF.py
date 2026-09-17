import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile

class ProtoAdaIN(nn.Module):
    """
    原型感知自适应实例归一化
    """
    def __init__(self, embed_dim, style_hidden=64):
        super().__init__()
        # 标准化层：不带仿射参数，由原型动态生成
        self.norm = nn.BatchNorm1d(embed_dim, affine=False)
        
        # 条件参数生成网络：输入原型嵌入，输出 γ, β, σ
        self.param_net = nn.Sequential(
            nn.Linear(embed_dim, style_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(style_hidden, embed_dim * 3)
        )
        
        # 初始化：让初始状态接近标准 BN，避免训练初期崩溃
        nn.init.zeros_(self.param_net[-1].weight)
        nn.init.zeros_(self.param_net[-1].bias)

    def forward(self, x, z_q):
        """
        Args:
            x:   [B, D, L] Encoder 输出的连续特征
            z_q: [B, D, L] VQ 量化后的原型特征 (prototype embeddings)
        Returns:
            out: [B, D, L] 风格解耦后的特征
        """
        B, D, L = x.shape
        
        # Step 1: 标准化（
        x_norm = self.norm(x)  # [B, D, L]
        
        # Step 2: 根据原型生成逐时间步的归一化参数
        # z_q permute → [B, L, D] → MLP → [B, L, 3*D]
        params = self.param_net(z_q.permute(0, 2, 1))
        gamma, beta, sigma = params.chunk(3, dim=-1)  # each: [B, L, D]
        
        # reshape 回 [B, D, L] 以匹配 x_norm
        gamma = gamma.permute(0, 2, 1)
        beta  = beta.permute(0, 2, 1)
        sigma = sigma.permute(0, 2, 1)
        
        # Step 3: 注入自适应噪声（仅训练时）
        if self.training:
            noise = torch.randn_like(x_norm) * F.softplus(sigma)
            out = gamma * (x_norm + noise) + beta
        else:
            # 推理时关闭噪声，仅做条件化仿射变换
            out = gamma * x_norm + beta
            
        return out

# ==========================================
# 1. 基础组件
# ==========================================

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(DepthwiseSeparableConv1d, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MobileNetV1_1D(nn.Module):
    def __init__(self, num_classes=2, in_channels=12, width_mult=1.0):
        super(MobileNetV1_1D, self).__init__()
        def scale_channels(c): return int(c * width_mult)

        self.stem = nn.Sequential(
            ConvBlock1D(in_channels, scale_channels(24), kernel_size=3, stride=2),
            ConvBlock1D(scale_channels(24), scale_channels(48), kernel_size=3, stride=2)
        )
        self.stage1 = nn.Sequential(
            ConvBlock1D(scale_channels(48), scale_channels(64), kernel_size=3, stride=2),
            ConvBlock1D(scale_channels(64), scale_channels(64), kernel_size=3),
            ConvBlock1D(scale_channels(64), scale_channels(64), kernel_size=3)
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv1d(scale_channels(64), scale_channels(96), kernel_size=3, stride=2),
            DepthwiseSeparableConv1d(scale_channels(96), scale_channels(96), kernel_size=3, stride=1)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d): nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x  # 输出 shape: (B, 96, 156) 输入长度2500


# ==========================================
# 2. 向量量化层
# ==========================================

class VectorQuantizer(nn.Module):
    """
    标准 VQ 层 (VQ-VAE 风格)
    包含 Codebook Loss 和 Commitment Loss，使用 Straight-Through Estimator (STE)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, 
        decay = 0.99,
        epsilon = 1e-5,
        diversity_weight = 0.01,
        dist_match_weight = 0.01,):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.diversity_weight = diversity_weight
        self.dist_match_weight = dist_match_weight
        
        # 码本初始化
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer('ema_count', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight', self.embeddings.weight.data.clone())

    def diversity_loss(self):
        """
        Codebook 多样性损失:计算所有 codebook 向量两两之间的欧氏距离，取平均。

        当 codebook 向量互相远离时（高多样性），loss 趋近于 0（好）。
        当 collapse 发生时（多个向量相同），loss 为很大的负数（坏）。

        训练时最小化 -dists.mean()，等价于最大化平均距离。
        """
        if self.num_embeddings <= 1:
            return torch.tensor(0.0, device=self.embeddings.weight.device)

        # torch.pdist 计算上三角矩阵的 pairwise L2 距离
        # 返回形状: [num_embeddings * (num_embeddings - 1) / 2]
        dists = torch.pdist(self.embeddings.weight, p=2)
        return -dists.mean()

    def distribution_matching_loss(
        self,
        z_flat,      # [B*L, D]  encoder 输出的扁平化特征
        z_q_flat,     # [B*L, D]  量化后的扁平化特征
    ):
        """
        分布匹配损失：对齐 encoder 输出 (z) 与量化输出 (z_q) 的均值和方差。

        这确保 codebook 整体覆盖 encoder 输出的分布范围，
        减少"encoder 输出在某个区域但 codebook 没有对应 code"的情况。
        """
        # 一阶矩（均值）对齐
        mu_z = z_flat.mean(dim=0)      # [D]
        mu_q = z_q_flat.mean(dim=0)    # [D]
        mean_loss = F.mse_loss(mu_z, mu_q)

        # 二阶矩（方差）对齐
        var_z = z_flat.var(dim=0, unbiased=False)    # [D]
        var_q = z_q_flat.var(dim=0, unbiased=False)  # [D]
        var_loss = F.mse_loss(var_z, var_q)

        return mean_loss + var_loss

    def forward(self, z):
        """
        Args:
            z: [B, D, L]  encoder 输出的连续特征图

        Returns:
            z_q_ste:      [B, D, L]  量化后的特征（STE 梯度回传）
            vq_loss:      scalar     总 VQ 损失（包含 commitment + codebook + diversity + dist_match）
            encoding_indices: [B, L]   每个位置对应的 code 索引
            similarities:   [B, L, K]  每个位置与每个 code 的负欧氏距离（用于后续分析）
        """
        B, D, L = z.shape

        # ---- 1. 形状变换: [B, D, L] → [B*L, D] ----
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)  # [B*L, D]

        # ---- 2. 计算欧氏距离（利用展开式优化） ----
        # ||z - e||^2 = ||z||^2 - 2*z·e + ||e||^2
        z_sq = z_flat.pow(2).sum(dim=1, keepdim=True)           # [B*L, 1]
        e_sq = self.embeddings.weight.pow(2).sum(dim=1, keepdim=True).t()  # [1, K]
        dist = z_sq - 2 * torch.matmul(z_flat, self.embeddings.weight.t()) + e_sq  # [B*L, K]

        # 负距离作为相似度（越大越相似）
        similarities = -dist.view(B, L, self.num_embeddings)  # [B, L, K]

        # ---- 3. 硬量化：最近邻查找 ----
        encoding_indices = torch.argmin(dist, dim=1)  # [B*L]
        z_q_flat = F.embedding(encoding_indices, self.embeddings.weight)  # [B*L, D]

        # 恢复形状
        z_q = z_q_flat.view(B, L, D).permute(0, 2, 1).contiguous()  # [B, D, L]
        encoding_indices = encoding_indices.view(B, L)  # [B, L]

        # ---- 4. 双向 Commitment Loss ----
        # Codebook Loss: 迫使 codebook 向量向 encoder 输出移动
        codebook_loss = F.mse_loss(z_q, z.detach())

        # Commitment Loss: 迫使 encoder 输出向 codebook 移动（防止 encoder 漂移）
        commitment_loss = F.mse_loss(z, z_q.detach())

        # ---- 5. 附加正则损失 ----
        # 多样性损失：防止 codebook collapse
        div_loss = self.diversity_loss()

        # 分布匹配损失：对齐统计量
        dm_loss = self.distribution_matching_loss(z_flat, z_q_flat)

        # ---- 6. 总 VQ 损失 ----
        vq_loss = (
            codebook_loss
            + self.commitment_cost * commitment_loss
            + self.diversity_weight * div_loss
            + self.dist_match_weight * dm_loss
        )

        # ---- 7. Straight-Through Estimator (STE) ----
        # 前向传播用量化后的 z_q，反向传播时梯度直接穿过 z
        z_q_ste = z + (z_q - z).detach()

        return z_q_ste, vq_loss, encoding_indices


class PrototypeAttentionClassifier(nn.Module):
    """
    原型驱动的分类头
    """
    def __init__(self, num_prototypes, embedding_dim, num_classes):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # 每个类别有一个可学习的 query
        self.class_queries = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=4, 
            dropout=0.1, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, 1, bias=False)  # 每个query输出一个logit
            
    
    def forward(self, z_q, encoding_indices):
        """
        Args:
            z_q: [B, D, L] 量化后的特征
            encoding_indices: [B, L] 每个时间步的code索引
        Returns:
            logits: [B, C]
            attn_weights: [B, C, K] 或 [B, C, L] 用于可视化解释
        """
        B, D, L = z_q.shape
            
        # z_q permute to [B, L, D] for attention
        z_q_seq = z_q.permute(0, 2, 1)  # [B, L, D]
        
        # class_queries expand to batch: [B, C, D]
        queries = self.class_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-Attention: queries attend to z_q sequence
        # attn_output: [B, C, D], attn_weights: [B, C, L]
        attn_output, attn_weights = self.cross_attn(
            query=queries, 
            key=z_q_seq, 
            value=z_q_seq,
            need_weights=True,
            average_attn_weights=False  # 保留所有head的权重用于可视化
        )
        
        attn_output = self.norm(attn_output)  # [B, C, D]
        logits = self.head(attn_output).squeeze(-1)  # [B, C]
            
        return logits, attn_weights


# ==========================================
# 3. 主模型：ProtoAF
# ==========================================

class ProtoAF(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_leads=12,
        width_mult=1.0,
        num_prototypes=128,      # 码本大小 (Codebook Size)
        commitment_cost=0.25,    # VQ 承诺损失权重 (beta)
        freeze_encoder=False     # 是否冻结预训练编码器
    ):
        super().__init__()
        
        # 1. 初始化 Encoder (只到 stage2)
        self.encoder = MobileNetV1_1D(num_classes=num_classes, in_channels=num_leads, width_mult=width_mult)
        
        # 获取 stage2 的输出通道数 (用于 VQ 的 embedding_dim)
        # 假设 width_mult=1.0 时，stage2 输出通道为 96
        self.embedding_dim = int(96 * width_mult) 
        
        # 2. 向量量化层
        self.quantizer = VectorQuantizer(
            num_embeddings=num_prototypes,
            embedding_dim=self.embedding_dim,
            commitment_cost=commitment_cost
        )

        self.proto_adain = ProtoAdaIN(embed_dim=self.embedding_dim)
        
        # 3. 分类头
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),  # (B, D, L) -> (B, D, 1)
        #     nn.Flatten(),             # (B, D, 1) -> (B, D)
        #     nn.Dropout(0.2),
        #     nn.Linear(self.embedding_dim, num_classes)
        # )
        self.classifier = PrototypeAttentionClassifier(
            num_prototypes=num_prototypes,
            embedding_dim=self.embedding_dim,
            num_classes=num_classes
        )
        
        # 4. 冻结编码器 (可选)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("⚠️ Encoder weights are FROZEN.")

    def forward(self, x, return_indices=False):
        """
        x: (B, 12, 2500)
        返回: logits, vq_loss, (可选) encoding_indices
        """
        # 1. 提取特征
        z = self.encoder(x)  # (B, 96, 156)
        
        # 2. 向量量化
        z_q, vq_loss, indices = self.quantizer(z)

        z_q = self.proto_adain(z, z_q)
        
        # 3. 分类
        # logits = self.classifier(z_q)
        logits, attn_weights = self.classifier(z_q, indices)
        
        if return_indices:
            return logits, vq_loss, indices, attn_weights
        return logits, vq_loss


import os
import time
import pandas as pd
import numpy as np

# ================= 配置区 =================
OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 1000      # 稳定性测试的样本数
NUM_WARMUP = 50         # 预热次数
BATCH_SIZE = 1          # 单样本推理 (模拟真实实时监测场景)
# ==========================================


def measure_per_sample_latency(model, device, num_samples=1000, num_warmup=50):
    """
    测量每个样本的独立推理延迟（用于评估稳定性/抖动）
    返回：包含每次推理耗时(ms)的列表
    """
    model.eval()
    model.to(device)
    
    # 生成单样本输入: (Batch=1, Channels=12, Length=2500)
    x = torch.randn(BATCH_SIZE, 12, 2500).to(device)
    
    latencies = []
    
    # 1. 预热阶段
    print(f"  🔥 Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # 2. 正式测量阶段
    print(f"  ⏱️  Measuring {num_samples} samples on {device}...")
    
    with torch.no_grad():
        for i in range(num_samples):
            if device.type == 'cuda':
                # GPU: 使用 CUDA Events 精确测量 GPU 端耗时
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = model(x)
                end_event.record()
                
                torch.cuda.synchronize()  # 等待GPU执行完毕
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                # CPU: 使用高精度计时器
                start_time = time.perf_counter()
                _ = model(x)
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000  # 转换为毫秒
                
            latencies.append(elapsed_ms)
            
            # 进度提示
            if (i + 1) % 200 == 0:
                print(f"    Processed {i + 1}/{num_samples} samples...")
                
    return latencies


def main():
    # 实例化模型
    print("🚀 Initializing MobileNet-V1-1D model...")
    model = ProtoAF(num_classes=2)
    x = torch.randn(1, 12, 2500)
    
    # ================= 1. 计算并保存 FLOPs & Params =================
    print("\n📊 Calculating model complexity...")
    flops, params = profile(model, inputs=(x,), verbose=False)
    
    flops_m = flops / 1e6   # 转换为 MFLOPs
    params_m = params / 1e6 # 转换为 M (百万参数)
    
    print(f"  ✅ FLOPs:  {flops_m:.2f} MFLOPs")
    print(f"  ✅ Params: {params_m:.2f} M")
    
    # 保存到 CSV
    complexity_df = pd.DataFrame({
        'Model': ['MobileNet-V1-1D'],
        'Params_M': [params_m],
        'FLOPs_MFLOPs': [flops_m],
        'Params_Raw': [int(params)],
        'FLOPs_Raw': [int(flops)]
    })
    complexity_csv = os.path.join(OUTPUT_DIR, "protoaf_complexity.csv")
    complexity_df.to_csv(complexity_csv, index=False)
    print(f"  💾 Saved to: {complexity_csv}")
    
    # ================= 2. 逐样本延迟稳定性测试 =================
    print("\n⏱️  Running per-sample latency stability test...")
    
    # --- CPU 测试 ---
    print("\n[CPU Test]")
    cpu_latencies = measure_per_sample_latency(
        model, 
        device=torch.device('cpu'), 
        num_samples=NUM_SAMPLES, 
        num_warmup=NUM_WARMUP
    )
    
    # --- GPU 测试 ---
    gpu_latencies = [np.nan] * NUM_SAMPLES  # 默认填充 NaN
    if torch.cuda.is_available():
        print("\n[GPU Test]")
        gpu_latencies = measure_per_sample_latency(
            model, 
            device=torch.device('cuda'), 
            num_samples=NUM_SAMPLES, 
            num_warmup=NUM_WARMUP
        )
    else:
        print("\n⚠️  CUDA not available. Skipping GPU latency test.")
    
    # 保存延迟数据到 CSV
    latency_df = pd.DataFrame({
        'sample_id': range(1, NUM_SAMPLES + 1),
        'cpu_latency_ms': cpu_latencies,
        'gpu_latency_ms': gpu_latencies
    })
    latency_csv = os.path.join(OUTPUT_DIR, "protoaf_latency.csv")
    latency_df.to_csv(latency_csv, index=False)
    print(f"\n💾 Latency data saved to: {latency_csv}")
    
    # ================= 3. 打印统计摘要 =================
    print("\n" + "="*50)
    print("📈 Latency Statistics (ms)")
    print("="*50)
    
    for device_name, latencies in [("CPU", cpu_latencies), ("GPU", gpu_latencies)]:
        if not np.isnan(latencies).all():
            arr = np.array(latencies)
            print(f"\n[{device_name}]")
            print(f"  Mean:     {np.mean(arr):.2f} ms")
            print(f"  Median:   {np.median(arr):.2f} ms")
            print(f"  Std Dev:  {np.std(arr):.2f} ms")
            print(f"  Min:      {np.min(arr):.2f} ms")
            print(f"  Max:      {np.max(arr):.2f} ms")
            print(f"  P95:      {np.percentile(arr, 95):.2f} ms")
            print(f"  P99:      {np.percentile(arr, 99):.2f} ms")
            print(f"  FPS:      {1000 / np.mean(arr):.2f}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()