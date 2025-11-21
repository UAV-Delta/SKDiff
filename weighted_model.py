import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.optim as optim
import copy

class SimpleConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=8):
        super(SimpleConvAutoencoder, self).__init__()
        # 编码部分（提取特征）
        self.conv1 = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, 3, padding=1)

        # 解码部分（重建输出）
        self.deconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 3, padding=1)
        self.deconv2 = nn.Conv1d(base_channels, in_channels, 3, padding=1)

    def forward(self, x):
        # 编码阶段
        x = torch.relu(self.conv1(x))   # [B, in_channels, L] -> [B, base_channels, L]
        x = torch.relu(self.conv2(x))   # [B, base_channels, L] -> [B, base_channels*2, L]

        # 解码阶段
        x = torch.relu(self.deconv1(x)) # [B, base_channels*2, L] -> [B, base_channels, L]
        x = self.deconv2(x)             # [B, base_channels, L] -> [B, in_channels, L]
        return x
'''
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        """
        dim: embedding dimension
        max_period: controls frequency range of sinusoids
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        t: tensor (B,) of floats or ints (time step indices)
        returns: (B, dim)
        """
        half = self.dim // 2
        device = t.device
        t = t.float()
        inv_freq = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=device).float() / half
        )  # (half,)
        sinusoid_in = t[:, None] * inv_freq[None, :]  # (B, half)
        emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)
'''
# ---------- Conditional UNet1D (保持不变) ----------

def sinusoidal_time_embedding(t, dim, max_period=10000):
    """
    t: tensor (B,) of floats or ints (time step indices)
    dim: embedding dimension (even is convenient)
    returns: (B, dim)
    """
    half = dim // 2
    # use float tensor for calculations
    device = t.device
    t = t.float()
    inv_freq = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=device).float() / half
    )  # (half,)
    # shape: (B, half)
    sinusoid_in = t[:, None] * inv_freq[None, :]
    emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


class ConditionalUNet1D(nn.Module):
    def __init__(self, station_emb_dim=16, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        # ---------- time embedding: sinusoidal -> MLP ----------
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # station embedding projector to same dim as time_emb
        self.station_mlp = nn.Sequential(
            nn.Linear(station_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ---------- simple encoder/decoder layers ----------
        self.conv1 = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, 3, padding=1)
        self.deconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 3, padding=1)
        self.deconv2 = nn.Conv1d(base_channels, in_channels, 3, padding=1)

        # ---------- per-layer linear projections from time+station emb to channel-bias ----------
        # these allow embedding to be added to each layer features without assuming dims equal
        self.to_bias_conv1 = nn.Linear(time_emb_dim, self.conv1.out_channels)
        self.to_bias_conv2 = nn.Linear(time_emb_dim, self.conv2.out_channels)
        self.to_bias_deconv1 = nn.Linear(time_emb_dim, self.deconv1.out_channels)
        # deconv2 outputs in_channels so projection not necessary, but keep for symmetry if needed
        self.to_bias_deconv2 = nn.Linear(time_emb_dim, self.deconv2.out_channels)

    def forward(self, x, t, station_emb):
        """
        x: (B, 1, L)
        t: (B,) long or float tensor with timestep indices
        station_emb: (B, station_emb_dim)
        """
        # 1) sinusoidal embed
        t_emb_sin = sinusoidal_time_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb_sin)  # (B, time_emb_dim)

        # 2) station embed
        s_emb = self.station_mlp(station_emb)  # (B, time_emb_dim)

        # 3) fuse
        ts_emb = t_emb + s_emb  # (B, time_emb_dim)

        # 4) project to layer-wise biases and add (broadcast over length dim)
        bias1 = self.to_bias_conv1(ts_emb).unsqueeze(-1)     # (B, C1, 1)
        bias2 = self.to_bias_conv2(ts_emb).unsqueeze(-1)     # (B, C2, 1)
        bias_d1 = self.to_bias_deconv1(ts_emb).unsqueeze(-1) # (B, C3, 1)
        bias_d2 = self.to_bias_deconv2(ts_emb).unsqueeze(-1) # (B, C4, 1)

        # encoder
        h1 = F.relu(self.conv1(x) + bias1)   # (B, C1, L)
        h2 = F.relu(self.conv2(h1) + bias2)  # (B, C2, L)

        # decoder
        u1 = F.relu(self.deconv1(h2) + bias_d1)  # (B, C1, L)  (deconv1 out_channels == base_channels)
        out = self.deconv2(u1 + bias_d2)         # (B, in_channels, L)
        return out



# ---------- Conditional DDPM1D（加入 station 权重到 loss） ----------
class ConditionalDDPM1D(nn.Module):
    def __init__(self, unet, seq_len=168, device='cuda', timesteps=1000):
        super().__init__()
        self.unet = unet
        self.seq_len = seq_len
        self.device = device
        self.timesteps = timesteps

        # beta schedule
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)  # (T,)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # (T,)

    def q_sample(self, x0, t, noise=None):
        """
        x0: (B,1,L)
        t: (B,) long tensor in [0, T-1]
        noise: same shape as x0
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # gather alpha_bar for each t
        a_bar = self.alpha_bar[t].to(x0.device)  # (B,)
        sqrt_a_bar = torch.sqrt(a_bar)[:, None, None]  # (B,1,1)
        sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar)[:, None, None]
        return sqrt_a_bar * x0 + sqrt_one_minus_a_bar * noise

    def forward(self, x0, station_emb, station_weight=None, normalize_weights=True):
        """
        x0: (B,1,L)    - clean data
        station_emb: (B, emb_dim)
        station_weight: None or (B,) or (B,1) or scalar
            - weight >= 0, 代表每个样本的损失重要性
        normalize_weights: bool - 是否按 weights.sum() 归一化（默认 True）
        返回: 标量 loss (tensor)
        """
        b = x0.shape[0]
        device = x0.device

        # 随机抽取时间步 t
        t = torch.randint(0, self.timesteps, (b,), device=device).long()  # (B,)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)  # (B,1,L)

        # unet 预测噪声
        predicted_noise = self.unet(x_t, t.float(), station_emb)  # (B,1,L)

        # 逐样本 MSE (在通道和时间维上求均值)
        per_sample_mse = torch.mean((predicted_noise - noise) ** 2, dim=[1,2])  # (B,)

        # 处理 station_weight
        if station_weight is None:
            weights = torch.ones_like(per_sample_mse, device=device)  # (B,)
        else:
            weights = station_weight.to(device).squeeze()  # (B,) or scalar
            if weights.dim() == 0:
                weights = weights.repeat(b)
            elif weights.numel() == b and weights.dim() == 1:
                pass
            else:
                # try to flatten (B,1) -> (B,)
                weights = weights.view(-1)[:b]

            # 保护：负值截断，避免异常
            weights = torch.clamp(weights, min=0.0)

        # 计算加权损失
        if normalize_weights:
            denom = weights.sum()
            # 防止除以 0
            if denom.item() == 0:
                denom = torch.tensor(1.0, device=device)
            loss = (per_sample_mse * weights).sum() / denom #保证loss不会因为权重大小改变数量级
        else:
            loss = (per_sample_mse * weights).mean()  # 或者直接 mean

        return loss

    @torch.no_grad()
    def sample(self, station_emb, batch_size=None):
        """
        station_emb: (B, emb_dim) or (batch_size, emb_dim)
        batch_size: optional, if None -> use station_emb.shape[0]
        returns: x generated (B,1,L)
        """
        if batch_size is None:
            batch_size = station_emb.shape[0]
        x = torch.randn(batch_size, 1, self.seq_len, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.float)
            eps = self.unet(x, t_batch, station_emb)
            alpha_t = self.alpha[t].to(x.device)
            alpha_bar_t = self.alpha_bar[t].to(x.device)
            beta_t = self.beta[t].to(x.device)

            # DDPM 反向一步（简化形式）
            x = 1.0 / torch.sqrt(alpha_t) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
        return x





