import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math

# -------------- CrossAttentionScoreWeights --------------
class CrossAttentionScoreWeights(nn.Module):
    """
    Multi-head cross-attention scoring module.

    station_embs: (B, M, d_model)
    target_emb:   (B, d_model)
    returns lambdas: (B, M)
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # 为接口保留（不用于传播）

    def forward(self, station_embs, target_emb):
        """
        station_embs: (B, M, d_model)
        target_emb:   (B, d_model)
        return: lambdas (B, M)
        """
        B, M, d = station_embs.shape
        assert d == self.d_model

        Q = self.w_q(target_emb)            # (B, d_model)
        K = self.w_k(station_embs)          # (B, M, d_model)
        V = self.w_v(station_embs)          # (B, M, d_model)  # 不用于传播

        # reshape to heads
        Q_h = Q.view(B, self.n_heads, self.d_head)                       # (B, H, d_head)
        K_h = K.view(B, M, self.n_heads, self.d_head).permute(0,2,1,3)    # (B, H, M, d_head)

        # compute raw scores: (B, H, M)
        Q_exp = Q_h.unsqueeze(2)                   # (B, H, 1, d_head)
        K_t = K_h.transpose(-2, -1)                # (B, H, d_head, M)
        raw_scores = torch.matmul(Q_exp, K_t).squeeze(2)  # (B, H, M)
        raw_scores = raw_scores / math.sqrt(self.d_head)

        # per-head softmax over M stations -> head_k
        head = F.softmax(raw_scores, dim=-1)  # (B, H, M)

        # cross-head fusion as you specified:
        head_soft = F.softmax(head, dim=-1)   # (B, H, M)
        weighted_heads = head_soft * head     # (B, H, M)
        lambdas = weighted_heads.sum(dim=1)   # (B, M)

        lambdas = torch.clamp(lambdas, min=0.0)
        return lambdas


# -------------- JS divergence for distributions --------------
def js_divergence(p, q, eps=1e-8, reduction='mean'):
    """
    p, q: (B, P) probability distributions (non-negative, sum to 1 across dim=1)
    returns: scalar loss (JS divergence)
    """
    # ensure numeric stability
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    m = 0.5 * (p + q)
    # KL divergence: sum p * log(p/m)
    kl_pm = torch.sum(p * torch.log(p / m), dim=1)
    kl_qm = torch.sum(q * torch.log(q / m), dim=1)
    js = 0.5 * (kl_pm + kl_qm)  # (B,)
    if reduction == 'mean':
        return js.mean()
    elif reduction == 'sum':
        return js.sum()
    else:
        return js  # return per-sample


# -------------- Meta-learning step（POI-based outer loss） --------------
def meta_train_step_with_poi(ddpm_model, weight_model, optimizer_pred, optimizer_weight,
                             x0, station_embs, target_emb,
                             station_pois, target_pois,
                             inner_steps=1, device='cuda',
                             normalize_lambda_for_poi=True):
    """
    ddpm_model: ConditionalDDPM1D (你的预测模型)
    weight_model: CrossAttentionScoreWeights
    x0: (B,1,L)
    station_embs: (B, M, emb_dim)
    target_emb: (B, emb_dim)
    station_pois: (B, M, P)  -- 每个 station 周围的 POI 分布（非负）。最好每个 station 的 POI 已归一化为分布。
    target_pois:  (B, P)     -- 目标城市的 POI 分布（每样本归一化）

    过程：
      1) outer: compute lambdas (no grad inside inner loop -> freeze weight model in inner loop)
      2) inner: use detached lambdas to build weighted embedding -> update ddpm_model by its own loss
      3) outer: recompute lambdas (with grad) -> aggregate POI distributions -> compute JS divergence with target_pois -> update weight_model
    """
    ddpm_model.train()
    weight_model.train()

    B, M, emb_dim = station_embs.shape
    _, _, P = station_pois.shape
    device = x0.device

    eps = 1e-8

    # ---------- 1) compute lambdas and freeze them for inner loop ----------
    with torch.no_grad():
        lambdas_detached = weight_model(station_embs, target_emb)  # (B, M), no grad
        # Normalize lambdas to sum to 1 per sample if needed for embedding aggregation
        if normalize_lambda_for_poi:
            denom = lambdas_detached.sum(dim=1, keepdim=True)  # (B,1)
            denom = torch.where(denom > 0, denom, torch.tensor(1.0, device=device))
            norm_lambdas_detached = lambdas_detached / denom
        else:
            norm_lambdas_detached = lambdas_detached

        # weighted embedding for inner ddpm update
        weighted_emb = torch.sum(station_embs * norm_lambdas_detached.unsqueeze(-1), dim=1)  # (B, emb_dim)

    # ---------- 2) inner loop: update ddpm_model (prediction model) ----------
    # note: we only update ddpm params here using ddpm_model's loss
    for _ in range(inner_steps):
        optimizer_pred.zero_grad()
        loss_pred = ddpm_model(x0, weighted_emb)  # 返回 scalar loss
        loss_pred.backward()
        optimizer_pred.step()

    # ---------- 3) outer loop: recompute lambdas (with grad) and update weight_model ----------
    optimizer_weight.zero_grad()
    lambdas = weight_model(station_embs, target_emb)  # (B, M)  <-- requires grad

    # Normalize lambdas when aggregating POIs so the aggregated vector is a distribution
    if normalize_lambda_for_poi:
        denom = lambdas.sum(dim=1, keepdim=True)  # (B,1)
        # avoid divide-by-zero
        denom = torch.where(denom > 0, denom, torch.tensor(eps, device=device))
        norm_lambdas = lambdas / denom
    else:
        norm_lambdas = lambdas

    # aggregate station POI distributions: weighted sum over M -> (B, P)
    agg_poi = torch.sum(station_pois * norm_lambdas.unsqueeze(-1), dim=1)  # (B, P)

    # ensure agg_poi and target_pois are valid distributions (non-neg and sum-to-1)
    agg_poi = agg_poi.clamp(min=eps)
    agg_poi = agg_poi / (agg_poi.sum(dim=1, keepdim=True) + eps)

    target_pois_clamped = target_pois.clamp(min=eps)
    target_pois_clamped = target_pois_clamped / (target_pois_clamped.sum(dim=1, keepdim=True) + eps)

    # compute JS divergence (outer loss) -- smaller is better
    loss_outer = js_divergence(agg_poi, target_pois_clamped, eps=eps, reduction='mean')

    # backprop only updates weight_model (optimizer_weight)
    loss_outer.backward()
    optimizer_weight.step()

    # return scalar losses for logging
    return {
        'loss_pred': loss_pred.item() if 'loss_pred' in locals() else None,
        'loss_outer_js': loss_outer.item()
    }


# -------------- 使用示例（Dummy 数据） --------------
if __name__ == "__main__":
    # 假设
    emb_dim = 64
    seq_len = 168
    num_stations = 5
    num_poi_types = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init models (注意：用你自己的 UNet 替换 Identity)
    weight_model = CrossAttentionScoreWeights(emb_dim, n_heads=4).to(device)
    unet = nn.Identity().to(device)   # 替换为你的 UNet
    from weighted_model import ConditionalDDPM1D  # 你原来的类（注意 import 路径）
    ddpm_model = ConditionalDDPM1D(unet, seq_len=seq_len, device=device)

    optimizer_pred = Adam(ddpm_model.parameters(), lr=1e-4)
    optimizer_weight = Adam(weight_model.parameters(), lr=1e-4)

    # dummy batch
    B = 8
    x0 = torch.randn(B,1,seq_len).to(device)
    station_embs = torch.randn(B, num_stations, emb_dim).to(device)
    target_emb = torch.randn(B, emb_dim).to(device)

    # station_pois: non-negative; for demo we sample and normalize per-station to make them distributions
    raw = torch.rand(B, num_stations, num_poi_types).to(device)
    station_pois = raw / (raw.sum(dim=2, keepdim=True) + 1e-8)  # each station a distribution over POI types

    # target_pois: distribution per sample
    raw_t = torch.rand(B, num_poi_types).to(device)
    target_pois = raw_t / (raw_t.sum(dim=1, keepdim=True) + 1e-8)

    stats = meta_train_step_with_poi(ddpm_model, weight_model, optimizer_pred, optimizer_weight,
                                     x0, station_embs, target_emb,
                                     station_pois, target_pois,
                                     inner_steps=1, device=device)

    print("loss_pred:", stats['loss_pred'], "loss_outer_js:", stats['loss_outer_js'])
