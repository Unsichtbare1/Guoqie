"""
麻将 AI 行为克隆训练脚本（Transformer 版）

与 train-topk.py 的区别：
- 模型骨干从 CNN+MLP 替换为 Transformer Encoder
- 把 34 种牌型当作长度为 34 的序列，每个位置是一个 1012 维 token
- 用 self-attention 建模任意两张牌之间的全局关系
  （CNN 受限于 3×1 感受野，无法直接捕捉跨花色对偶关系）

保持一致的部分：
- 数据集 / 数据划分（按局划分）
- Top-k 采样与评估指标

默认配置：
- readout=per_tile（逐牌位读出头），可用 --readout pooled 回到旧路径
- 纯 CE 损失（soft_alpha=0、entropy_weight=0）
- epochs=20、warmup_steps=400、CosineAnnealingLR(eta_min=1e-5)

用法:
    python train/train-transformer.py --data "data/output/huolongguo.npz" --save "models/huolongguo_transformer.pt"
    python train/train-transformer.py --data "data/output/huolongguo.npz" --epochs 5 --d-model 384 --num-layers 4
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. 数据集（与 train-topk.py 完全一致）
# ============================================================
class MahjongDataset(Dataset):
    """
    读取 export_decision_points.py 导出的 .npz 文件。
    每个样本返回 (obs, mask, action)。
    """
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.obs = data["obs"]          # (N, 1012, 34) float32
        self.masks = data["masks"]      # (N, 46) bool
        self.action = data["action"]    # (N,) int64
        self.game_index = data["game_index"]
        self.file_index = data["file_index"]

        self.game_keys = list(zip(
            self.file_index.tolist(),
            self.game_index.tolist(),
        ))
        assert len(self.obs) == len(self.masks) == len(self.action)

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]).float(),
            torch.from_numpy(self.masks[idx]).bool(),
            torch.tensor(self.action[idx], dtype=torch.long),
        )


def split_by_game(dataset: MahjongDataset, val_ratio: float = 0.1, seed: int = 42):
    """按局划分训练/验证集，避免数据泄漏。"""
    unique_games = sorted(set(dataset.game_keys))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_games)
    n_val = max(1, int(len(unique_games) * val_ratio))
    val_games = set(unique_games[:n_val])
    train_games = set(unique_games[n_val:])
    train_idx = [i for i, k in enumerate(dataset.game_keys) if k in train_games]
    val_idx = [i for i, k in enumerate(dataset.game_keys) if k in val_games]
    return train_idx, val_idx


# ============================================================
# 2. Transformer 模型
# ============================================================
class PositionalEncoding(nn.Module):
    """正弦余弦位置编码，给 34 个牌型位置注入顺序信息。"""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class MahjongTransformer(nn.Module):
    """
    输入: (B, 1012, 34)   1012 特征通道 × 34 牌型位置
    输出: (B, 46)         动作 logits

    设计思路:
    - 把 34 个牌型位置当作长度为 34 的序列，每个位置是 1012 维 token
    - Linear(1012 -> d_model) 做 token embedding
    - 加正弦位置编码注入牌型顺序信息
    - Transformer Encoder 用 self-attention 建模任意两张牌的全局关系
      （如三色同顺、字牌跨花色对偶等 CNN 难以捕捉的非局部模式）
    - readout="pooled"：mean pooling 后经 MLP 头输出 46 维动作（旧路径，保留对照）
    - readout="per_tile"：逐牌位读出头
        切牌 0–33  ← 每个牌位共享 Linear(d_model→1)
        赤 5 34–36 ← 5m/5p/5s 三个牌位接 Linear(d_model→3)
        非切牌 37–45 ← mean pooling 后接 Linear(d_model→9)
    """

    def __init__(
        self,
        in_channels: int = 1012,
        num_actions: int = 46,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        seq_len: int = 34,
        readout: str = "per_tile",
    ):
        super().__init__()
        # token embedding：把每个位置的 1012 维特征投影到 d_model
        self.token_proj = nn.Linear(in_channels, d_model)
        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.emb_dropout = nn.Dropout(dropout)
        # Transformer Encoder（batch_first=True，输入 (B, L, D)）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 读出头：两种实现二选一，避免把未使用的头算进参数量
        assert readout in ("pooled", "per_tile")
        self.readout = readout
        if readout == "pooled":
            # 旧路径：mean pooling 后经 MLP 输出 46 维
            self.head = nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, num_actions),
            )
        else:
            # 逐牌位：34 个切牌 + 3 个赤 5 + 9 个非切牌
            # 动作 0–33 与 obs 的 34 个牌型位置一一对应
            self.tile_head = nn.Linear(d_model, 1)
            self.aka_head = nn.Linear(d_model, 3)
            self.other_head = nn.Linear(d_model, 9)

    def forward(self, x):
        # x: (B, 1012, 34) -> (B, 34, 1012)
        x = x.transpose(1, 2)
        # token embedding
        x = self.token_proj(x)            # (B, 34, d_model)
        x = self.pos_enc(x)               # 加位置编码
        x = self.emb_dropout(x)
        # Transformer Encoder
        x = self.transformer(x)           # (B, 34, d_model)
        x = self.norm(x)

        if self.readout == "pooled":
            # 旧路径：mean pooling over 34 positions
            x = x.mean(dim=1)             # (B, d_model)
            return self.head(x)           # (B, 46)

        # 逐牌位读出头：动作索引与 ACTION_NAMES 保持一致
        # 0–33 切牌（每个牌位一个 logit）
        tile_logits = self.tile_head(x).squeeze(-1)        # (B, 34)
        # 34–36 赤 5（5m/5p/5s 对应牌位 4/13/22）
        # 三行各自经 Linear(d→3)，取对角线：第 i 行只提供第 i 个赤 5 的 logit
        aka_scores = self.aka_head(x[:, [4, 13, 22]])      # (B, 3, 3)
        aka_logits = aka_scores.diagonal(dim1=1, dim2=2)   # (B, 3)
        # 37–45 非切牌（riichi/chi/pon/kan/hora/ryukyoku/pass）
        other_logits = self.other_head(x.mean(dim=1))      # (B, 9)
        return torch.cat([tile_logits, aka_logits, other_logits], dim=-1)


# ============================================================
# 3. Top-k 损失函数（与 train-topk.py 完全一致）
# ============================================================
NEG_INF = torch.finfo(torch.float32).min


def topk_soft_loss(
    logits: torch.Tensor,
    action: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    alpha: float,
    temperature: float = 1.0,
    entropy_weight: float = 0.0,
):
    """Top-k 软标签 + 熵正则化损失，与 train-topk.py 完全一致。"""
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    log_probs = F.log_softmax(masked_logits, dim=-1)

    ce_loss = F.nll_loss(log_probs, action)

    if alpha > 0 and k > 1:
        with torch.no_grad():
            probs = torch.softmax(masked_logits / temperature, dim=-1)
            topk_probs, topk_idx = probs.topk(k, dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            soft_dist = torch.zeros_like(probs)
            soft_dist.scatter_(1, topk_idx, topk_probs)

        one_hot = F.one_hot(action, num_classes=46).float()
        target = (1 - alpha) * one_hot + alpha * soft_dist
        soft_ce = -(target * log_probs).sum(dim=-1).mean()
        loss = soft_ce
    else:
        loss = ce_loss

    if entropy_weight > 0:
        probs_for_entropy = torch.exp(log_probs)
        entropy = -(probs_for_entropy * log_probs).sum(dim=-1)
        loss = loss - entropy_weight * entropy.mean()

    return loss


# ============================================================
# 4. Top-k 采样（与 train-topk.py 完全一致）
# ============================================================
def topk_sample(
    logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    temperature: float = 1.0,
):
    """从合法动作的 top-k 中按概率采样。"""
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    topk_logits, topk_idx = masked_logits.topk(k, dim=-1)
    topk_probs = F.softmax(topk_logits / temperature, dim=-1)
    sampled = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
    return topk_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


# ============================================================
# 5. 支路尺度监控（只观察，不参与损失/选模）
# ============================================================
BRANCH_RANGES = (("tile", 0, 34), ("aka", 34, 37), ("other", 37, 46))


def init_monitor():
    return {
        "logit": {name: [0.0, 0.0, 0] for name, _, _ in BRANCH_RANGES},
        "branch": {"tile": 0, "aka": 0, "other": 0},
        "actions": torch.zeros(46, dtype=torch.long),
    }


def branch_monitor(logits, mask):
    """
    统计三条读出支路的原始 logit 均值/标准差，以及 masked argmax 的落点。
    logit 尺度失衡时，46 个动作共享的 masked softmax 会系统性偏向某一支路，
    这属于实现问题而非策略差异，故单独监控。
    """
    raw = logits.detach().double()
    pred = logits.masked_fill(~mask, NEG_INF).argmax(dim=-1)
    logit = {}
    for name, lo, hi in BRANCH_RANGES:
        seg = raw[:, lo:hi]
        logit[name] = (seg.sum().item(), (seg * seg).sum().item(), seg.numel())
    branch = {
        "tile": int((pred < 34).sum()),
        "aka": int(((pred >= 34) & (pred < 37)).sum()),
        "other": int((pred >= 37).sum()),
    }
    return logit, branch, torch.bincount(pred, minlength=46)


def merge_monitor(acc, mon):
    logit, branch, actions = mon
    for name, (s, ss, n) in logit.items():
        acc["logit"][name][0] += s
        acc["logit"][name][1] += ss
        acc["logit"][name][2] += n
    for k in acc["branch"]:
        acc["branch"][k] += branch[k]
    acc["actions"] += actions.cpu()


def format_monitor(acc):
    """支路 logit 尺度 + argmax 支路占比 + argmax 动作频率（%）。"""
    parts = []
    for name, _, _ in BRANCH_RANGES:
        s, ss, n = acc["logit"][name]
        if n:
            mean = s / n
            std = max(0.0, ss / n - mean * mean) ** 0.5
            parts.append(f"{name} μ{mean:+.3f} σ{std:.3f}")
    total = sum(acc["branch"].values())
    if total:
        bparts = " / ".join(
            f"{k} {acc['branch'][k] / total:.3f}"
            for k in ("tile", "aka", "other"))
        aparts = " ".join(
            f"{i}:{c / total * 100:.1f}%"
            for i, c in enumerate(acc["actions"].tolist()) if c)
    else:
        bparts, aparts = "-", "-"
    return (f"logit 尺度[{' | '.join(parts)}] | "
            f"argmax 支路[{bparts}] | argmax 动作[{aparts}]")


# ============================================================
# 6. 评估（与 train-topk.py 完全一致）
# ============================================================
def evaluate(model, loader, device, top_k=3, sample_times=10):
    """
    返回 loss / top1 / top1(n>=2) / top2 / top3 / sampled_acc / NCE,
    以及验证集合法动作数直方图。
    """
    model.eval()
    total_loss = 0.0
    total = 0
    total_n2 = 0
    topk_hits = {1: 0, 2: 0, 3: 0}
    top1_hits_n2 = 0
    sampled_hits = 0
    sampled_total = 0
    nce_sum = 0.0
    nce_count = 0
    # 困惑度用几何平均：PPL = exp(mean CE)。先累计桶内 CE 之和与样本数，
    # 最后再取指数；这样不会被少数 CE 极大的样本支配。
    ce_sum_n2 = 0.0
    ce_buckets = {"2": [0.0, 0], "3-5": [0.0, 0],
                  "6-10": [0.0, 0], ">=11": [0.0, 0]}
    conf_sum = 0.0
    conf_count = 0
    ece_bins = 15
    ece_counts = [0] * ece_bins
    ece_conf_sums = [0.0] * ece_bins
    ece_correct_sums = [0.0] * ece_bins
    n_hist: dict[int, int] = {}
    monitor_acc = init_monitor()

    with torch.no_grad():
        for obs, mask, action in loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)
            logits = model(obs)
            masked_logits = logits.masked_fill(~mask, NEG_INF)
            merge_monitor(monitor_acc, branch_monitor(logits, mask))

            loss = F.cross_entropy(masked_logits, action)
            total_loss += loss.item() * action.size(0)

            n_legal = mask.sum(dim=1)
            for v in n_legal.tolist():
                n_hist[v] = n_hist.get(v, 0) + 1
            n2 = n_legal >= 2

            for k in topk_hits:
                topk_pred = masked_logits.topk(k, dim=-1).indices
                hit = (topk_pred == action.unsqueeze(-1)).any(dim=-1)
                topk_hits[k] += hit.sum().item()
                if k == 1:
                    total_n2 += n2.sum().item()
                    top1_hits_n2 += hit[n2].sum().item()

            for _ in range(sample_times):
                sampled = topk_sample(logits, mask, top_k, temperature=1.0)
                sampled_hits += (sampled == action).sum().item()
                sampled_total += action.size(0)

            # NCE：合法集合内 softmax（logsumexp 保证数值稳定），
            # 逐样本计算后取平均；n==1 的样本没有选择空间，排除。
            lse = torch.logsumexp(masked_logits, dim=-1)
            ce_i = lse - masked_logits.gather(1, action.unsqueeze(1)).squeeze(1)
            nce_i = 1.0 - ce_i[n2] / torch.log(n_legal[n2].float())
            nce_sum += nce_i.sum().item()
            nce_count += n2.sum().item()

            # PPL 用几何平均：累计逐样本 CE，最后 exp(CE 均值)。
            # CE 用 float64，避免极端 logit 差距带来的精度损失。
            ce_n2 = ce_i[n2].double()
            n_n2 = n_legal[n2]
            ce_sum_n2 += ce_n2.sum().item()
            for name, lo, hi in (("2", 2, 2), ("3-5", 3, 5),
                                 ("6-10", 6, 10), (">=11", 11, 10**9)):
                sel = (n_n2 >= lo) & (n_n2 <= hi)
                if sel.any():
                    ce_buckets[name][0] += ce_n2[sel].sum().item()
                    ce_buckets[name][1] += int(sel.sum())

            # 置信度：合法集 masked softmax 的最大概率，仅在 n>=2 上统计。
            # ECE 用 15 个等宽区间，准确性为 top1。
            correct = masked_logits.argmax(dim=-1) == action
            conf_n2 = torch.softmax(masked_logits, dim=-1).max(dim=-1).values[n2]
            corr_n2 = correct[n2].double()
            conf_sum += conf_n2.sum().item()
            conf_count += conf_n2.numel()
            bin_idx = (conf_n2 * ece_bins).long().clamp(max=ece_bins - 1)
            b_cnt = torch.bincount(bin_idx, minlength=ece_bins).tolist()
            b_conf = torch.bincount(bin_idx, weights=conf_n2,
                                    minlength=ece_bins).tolist()
            b_corr = torch.bincount(bin_idx, weights=corr_n2,
                                    minlength=ece_bins).tolist()
            for b in range(ece_bins):
                ece_counts[b] += b_cnt[b]
                ece_conf_sums[b] += b_conf[b]
                ece_correct_sums[b] += b_corr[b]

            total += action.size(0)

    mean_conf = conf_sum / conf_count if conf_count else float("nan")
    ece = 0.0
    for b in range(ece_bins):
        if ece_counts[b]:
            acc_b = ece_correct_sums[b] / ece_counts[b]
            conf_b = ece_conf_sums[b] / ece_counts[b]
            ece += ece_counts[b] / conf_count * abs(acc_b - conf_b)

    return {
        "loss": total_loss / total,
        "top1_acc": topk_hits[1] / total,
        "top1_acc_n2": top1_hits_n2 / total_n2 if total_n2 else float("nan"),
        "top2_acc": topk_hits[2] / total,
        "top3_acc": topk_hits[3] / total,
        "sampled_acc": sampled_hits / sampled_total,
        "nce": nce_sum / nce_count if nce_count else float("nan"),
        "ppl": math.exp(ce_sum_n2 / nce_count) if nce_count else float("nan"),
        "ppl_buckets": {
            name: {
                "ppl": math.exp(v[0] / v[1]) if v[1] else float("nan"),
                "n": v[1],
            }
            for name, v in ce_buckets.items()
        },
        "mean_conf": mean_conf,
        "conf_gap": mean_conf - (top1_hits_n2 / total_n2 if total_n2 else float("nan")),
        "ece": ece,
        "n_legal_hist": n_hist,
        "monitor": monitor_acc,
    }


def setup_logging(log_path: str | None):
    """同时写文件与 stdout；每条记录立即 flush，避免缓冲/重定向丢行。"""
    logger = logging.getLogger("train_transformer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)

    def log(msg: str):
        logger.info(msg)
        for h in logger.handlers:
            h.flush()

    log(f"日志文件: {log_path}" if log_path else "日志仅输出到 stdout")
    return log


# ============================================================
# 6. 绘图
# ============================================================
def plot_losses(train_losses, val_losses, acc_curves, save_path="results/loss_curve_transformer.png"):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(epochs, train_losses, label="Training loss")
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Transformer - Loss")

    for name, vals in acc_curves.items():
        ax2.plot(epochs, vals, label=name)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Transformer - Accuracy Metrics")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"曲线已保存到: {save_path}")
    plt.show()


# ============================================================
# 7. 训练
# ============================================================
def train(args):
    log = setup_logging(args.log)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"使用设备: {device} | 随机种子: {args.seed}")
    log(f"模型: Transformer (d_model={args.d_model}, nhead={args.nhead}, "
        f"layers={args.num_layers}, dim_ff={args.dim_ff}, readout={args.readout})")
    log(f"Top-k: {args.top_k}, soft_alpha: {args.soft_alpha}, "
        f"entropy_weight: {args.entropy_weight}, temperature: {args.temperature}")
    log(f"训练预算: epochs={args.epochs}, warmup_steps={args.warmup_steps}, "
        f"lr={args.lr}, eta_min=1e-5")

    # 数据集
    dataset = MahjongDataset(args.data)
    log(f"总样本数: {len(dataset)}")

    train_idx, val_idx = split_by_game(dataset, val_ratio=args.val_ratio)
    log(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    in_channels = dataset.obs.shape[1]
    model = MahjongTransformer(
        in_channels=in_channels,
        num_actions=46,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        readout=args.readout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 线性 warmup + cosine 衰减（峰值 lr 不变，末尾 eta_min=1e-5）。
    # 按 batch（而非 epoch）步进，总步数 = epochs × 每轮 batch 数。
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = min(args.warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=max(1, warmup_steps)),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5),
        ],
        milestones=[warmup_steps],
    )
    log(f"总训练步数: {total_steps}（warmup {warmup_steps}）")

    best_val_acc = 0.0
    epochs_no_improve = 0
    save_path = Path(args.save)

    train_losses: list[float] = []
    val_losses: list[float] = []
    acc_curves: dict[str, list[float]] = {
        "top1": [], "top2": [], "top3": [], "sampled": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        train_mon = init_monitor()
        epoch_start = time.time()
        for obs, mask, action in train_loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)

            logits = model(obs)
            merge_monitor(train_mon, branch_monitor(logits, mask))
            loss = topk_soft_loss(
                logits, action, mask,
                k=args.top_k,
                alpha=args.soft_alpha,
                temperature=args.temperature,
                entropy_weight=args.entropy_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * action.size(0)
            total += action.size(0)

        train_loss = total_loss / total
        train_seconds = time.time() - epoch_start
        log(f"Epoch {epoch:3d} train 监控 | " + format_monitor(train_mon))
        metrics = evaluate(model, val_loader, device, top_k=args.top_k)
        epoch_seconds = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(metrics["loss"])
        acc_curves["top1"].append(metrics["top1_acc"])
        acc_curves["top2"].append(metrics["top2_acc"])
        acc_curves["top3"].append(metrics["top3_acc"])
        acc_curves["sampled"].append(metrics["sampled_acc"])

        log(
            f"Epoch {epoch:3d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"top1 {metrics['top1_acc']:.4f} | "
            f"top1(n>=2) {metrics['top1_acc_n2']:.4f} | "
            f"top2 {metrics['top2_acc']:.4f} | "
            f"top3 {metrics['top3_acc']:.4f} | "
            f"sampled {metrics['sampled_acc']:.4f} | "
            f"NCE {metrics['nce']:.4f} | "
            f"conf {metrics['mean_conf']:.4f} | gap {metrics['conf_gap']:+.4f} | "
            f"ECE {metrics['ece']:.4f} | "
            f"train {train_seconds:.1f}s | total {epoch_seconds:.1f}s"
        )
        log(f"Epoch {epoch:3d} val 监控 | " + format_monitor(metrics['monitor']))
        buckets = metrics["ppl_buckets"]
        log("PPL 分桶: " + " | ".join(
            f"{name} PPL {buckets[name]['ppl']:.3f}"
            f" (n={buckets[name]['n']})"
            for name in ("2", "3-5", "6-10", ">=11")))
        if epoch == 1:
            hist = metrics["n_legal_hist"]
            log("验证集合法动作数分布: " + " ".join(
                f"{k}:{hist[k]}" for k in sorted(hist)))

        if metrics["top1_acc"] > best_val_acc:
            best_val_acc = metrics["top1_acc"]
            torch.save({
                "model_state": model.state_dict(),
                "in_channels": in_channels,
                "num_actions": 46,
                "epoch": epoch,
                "val_acc": metrics["top1_acc"],
                "val_top2_acc": metrics["top2_acc"],
                "val_top3_acc": metrics["top3_acc"],
                "val_sampled_acc": metrics["sampled_acc"],
                "top_k": args.top_k,
                "soft_alpha": args.soft_alpha,
                "entropy_weight": args.entropy_weight,
                "temperature": args.temperature,
                "arch": "transformer",
                "readout": args.readout,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_ff": args.dim_ff,
                "n_params": n_params,
            }, save_path)
            log(f"  → 保存最优模型 (top1={metrics['top1_acc']:.4f}, "
                f"top2={metrics['top2_acc']:.4f}, top3={metrics['top3_acc']:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if (args.early_stop_patience > 0
                and epochs_no_improve >= args.early_stop_patience):
            log(f"早停：连续 {epochs_no_improve} 轮 top1 未提升（patience="
                f"{args.early_stop_patience}）")
            break

    log(f"\n训练完成。最优 top-1 准确率: {best_val_acc:.4f}")
    log(f"模型已保存到: {save_path}")
    plot_losses(train_losses, val_losses, acc_curves, save_path=args.curve)


# ============================================================
# 8. 命令行入口
# ============================================================
def build_argparser():
    parser = argparse.ArgumentParser(description="Transformer 版行为克隆训练脚本")
    parser.add_argument("--data", required=True, help=".npz 文件路径")
    parser.add_argument("--save", default="models/bc_model_transformer.pt", help="模型保存路径")
    parser.add_argument("--curve", default="results/loss_curve_transformer.png",
                        help="训练曲线保存路径")
    parser.add_argument("--log", default=None, help="日志文件路径（同时输出到 stdout）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（固定可复现）")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="连续多少轮 top1 无提升则早停；0 表示不早停")

    # Transformer 结构参数
    parser.add_argument("--d-model", type=int, default=384, help="token 维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=4, help="Encoder 层数")
    parser.add_argument("--dim-ff", type=int, default=1024, help="FFN 中间维")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=400, help="线性 warmup 步数")
    parser.add_argument("--readout", choices=["pooled", "per_tile"], default="per_tile",
                        help="读出头：pooled=旧路径，per_tile=逐牌位")

    # Top-k 采样参数（与 train-topk.py 一致）
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--soft-alpha", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    return parser


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
