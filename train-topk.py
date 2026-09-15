"""
麻将 AI 行为克隆训练脚本（Top-k 采样版）

相比 train.py 的改进：
1. Top-k 软标签损失：不再用纯 one-hot 目标，而是将 (1-α) 的概率给 GT
   动作、α 的概率按模型自身 top-k 预测分布分散到其他动作上。当几
   张牌概率相近时，模型不会被过度惩罚"选了另一张等价牌"。
2. 熵正则化：对合法动作的预测分布加 entropy reward，防止模型对
   单一动作过度自信、在等价牌之间无法区分。
3. Top-k 评估：报告 top-1 / top-2 / top-3 准确率，衡量"GT 是否在
   模型最看好的 k 个动作内"，比单纯 argmax 更能反映真实水平。
4. Top-k 采样准确率：从 top-k 合法动作中按概率采样，统计与 GT
   的命中率，模拟实际对局中的随机性。

用法:
    python train-topk.py --data "output/huolongguo.npz" --save "huolongguo_topk.pt"
    python train-topk.py --data "output/huolongguo.npz" --top-k 3 --soft-alpha 0.15 --entropy-weight 0.01
"""
from __future__ import annotations

import argparse
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
# 1. 数据集（与 train.py 完全一致）
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
# 2. 模型（与 train.py 完全一致）
# ============================================================
class MahjongNet(nn.Module):
    def __init__(self, in_channels: int = 1012, num_actions: int = 46, hidden: int = 256):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, 128, kernel_size=1)
        self.conv1 = nn.Conv1d(128, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 34, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x)


# ============================================================
# 3. Top-k 损失函数
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
    """
    Top-k 软标签 + 熵正则化损失。

    核心思路：
    - 标准 cross-entropy 用 one-hot 目标，当几张牌概率相近时，
      模型选了"另一张等价牌"会被严重惩罚，导致训练瓶颈。
    - Top-k 软标签：目标 = (1-α) * one_hot(GT) + α * top_k_dist
      其中 top_k_dist 是模型自身在合法动作上的 top-k 概率分布
      （detach，不参与梯度）。这样模型不会被强迫只押注一张牌，
      概率相近的牌可以分到部分概率质量。
    - 熵正则化：对合法动作的 softmax 分布加 -entropy 奖励，
      鼓励模型在不确定时保持分散而非坍缩到单一动作。

    参数:
        logits:      (B, 46) 原始 logits
        action:      (B,)    GT 动作 ID
        mask:        (B, 46) 合法动作掩码
        k:           top-k 的 k
        alpha:       软标签中 top-k 分布的权重 (0~1)
        temperature: top-k 分布的温度（越高越平滑）
        entropy_weight: 熵正则化权重（越大越鼓励分散）
    """
    # 掩码非法动作
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    log_probs = F.log_softmax(masked_logits, dim=-1)       # (B, 46)

    # --- 标准 cross-entropy（硬目标部分） ---
    ce_loss = F.nll_loss(log_probs, action)

    # --- Top-k 软标签部分 ---
    if alpha > 0 and k > 1:
        with torch.no_grad():
            probs = torch.softmax(masked_logits / temperature, dim=-1)  # (B, 46)
            # 取 top-k 合法动作的概率和索引
            topk_probs, topk_idx = probs.topk(k, dim=-1)                # (B, k)
            # 重归一化
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            # 构造 top-k 软分布（仅在 top-k 位置有值）
            soft_dist = torch.zeros_like(probs)
            soft_dist.scatter_(1, topk_idx, topk_probs)

        # 混合目标: (1-α) * one-hot(GT) + α * top-k 分布
        one_hot = F.one_hot(action, num_classes=46).float()  # (B, 46)
        target = (1 - alpha) * one_hot + alpha * soft_dist

        # 软交叉熵: -sum(target * log_prob)
        soft_ce = -(target * log_probs).sum(dim=-1).mean()

        # 最终损失 = 软交叉熵（已包含硬目标） + 熵正则化
        loss = soft_ce
    else:
        loss = ce_loss

    # --- 熵正则化 ---
    if entropy_weight > 0:
        # 对合法动作分布的熵加奖励（负号：熵越大损失越小）
        probs_for_entropy = torch.exp(log_probs)
        entropy = -(probs_for_entropy * log_probs).sum(dim=-1)  # (B,)
        # 只对有多个合法动作的样本计算（单个合法动作的熵为 0，无需正则化）
        loss = loss - entropy_weight * entropy.mean()

    return loss


# ============================================================
# 4. Top-k 采样
# ============================================================
def topk_sample(
    logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    temperature: float = 1.0,
):
    """
    从合法动作的 top-k 中按概率采样。

    参数:
        logits:     (B, 46)
        mask:       (B, 46) bool
        k:          top-k 的 k
        temperature: 采样温度（越高越接近均匀，越低越接近 argmax）
    返回:
        sampled: (B,) 采样到的动作 ID
    """
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    # 取 top-k
    topk_logits, topk_idx = masked_logits.topk(k, dim=-1)  # (B, k)
    # 按温度缩放后 softmax
    topk_probs = F.softmax(topk_logits / temperature, dim=-1)  # (B, k)
    # 从 top-k 中采样
    sampled = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)  # (B,)
    return topk_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # (B,)


# ============================================================
# 5. 评估
# ============================================================
def evaluate(model, loader, device, top_k=3, sample_times=10):
    """
    评估模型，返回多个指标：
    - loss:       掩码交叉熵损失
    - top1_acc:   贪心 argmax 准确率
    - top2_acc:   GT 在 top-2 内的比例
    - top3_acc:   GT 在 top-3 内的比例
    - sampled_acc: 从 top-k 采样 sample_times 次后的平均命中率
    """
    model.eval()
    total_loss = 0.0
    total = 0
    topk_hits = {1: 0, 2: 0, 3: 0}
    sampled_hits = 0
    sampled_total = 0

    with torch.no_grad():
        for obs, mask, action in loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)
            logits = model(obs)
            masked_logits = logits.masked_fill(~mask, NEG_INF)

            # 损失
            loss = F.cross_entropy(masked_logits, action)
            total_loss += loss.item() * action.size(0)

            # Top-k 准确率：GT 是否在 top-k 内
            for k in topk_hits:
                topk_pred = masked_logits.topk(k, dim=-1).indices  # (B, k)
                hit = (topk_pred == action.unsqueeze(-1)).any(dim=-1)
                topk_hits[k] += hit.sum().item()

            # Top-k 采样准确率
            for _ in range(sample_times):
                sampled = topk_sample(logits, mask, top_k, temperature=1.0)
                sampled_hits += (sampled == action).sum().item()
                sampled_total += action.size(0)

            total += action.size(0)

    return {
        "loss": total_loss / total,
        "top1_acc": topk_hits[1] / total,
        "top2_acc": topk_hits[2] / total,
        "top3_acc": topk_hits[3] / total,
        "sampled_acc": sampled_hits / sampled_total,
    }


# ============================================================
# 6. 绘图
# ============================================================
def plot_losses(train_losses, val_losses, acc_curves, save_path="loss_curve_topk.png"):
    """
    train_losses:  list[float]
    val_losses:    list[float]
    acc_curves:    dict[str, list[float]]  各准确率曲线
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(epochs, train_losses, label="Training loss")
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    for name, vals in acc_curves.items():
        ax2.plot(epochs, vals, label=name)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy Metrics")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"曲线已保存到: {save_path}")
    plt.show()


# ============================================================
# 7. 训练
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"Top-k: {args.top_k}, soft_alpha: {args.soft_alpha}, "
          f"entropy_weight: {args.entropy_weight}, temperature: {args.temperature}")

    # 数据集
    dataset = MahjongDataset(args.data)
    print(f"总样本数: {len(dataset)}")

    train_idx, val_idx = split_by_game(dataset, val_ratio=args.val_ratio)
    print(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    in_channels = dataset.obs.shape[1]
    model = MahjongNet(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    save_path = Path(args.save)

    # 记录训练和验证指标
    train_losses: list[float] = []
    val_losses: list[float] = []
    acc_curves: dict[str, list[float]] = {
        "top1": [], "top2": [], "top3": [], "sampled": [],
    }

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        total = 0
        for obs, mask, action in train_loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)

            logits = model(obs)
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

            total_loss += loss.item() * action.size(0)
            total += action.size(0)

        scheduler.step()

        # ---- 验证 ----
        train_loss = total_loss / total
        metrics = evaluate(model, val_loader, device, top_k=args.top_k)

        train_losses.append(train_loss)
        val_losses.append(metrics["loss"])
        acc_curves["top1"].append(metrics["top1_acc"])
        acc_curves["top2"].append(metrics["top2_acc"])
        acc_curves["top3"].append(metrics["top3_acc"])
        acc_curves["sampled"].append(metrics["sampled_acc"])

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"top1 {metrics['top1_acc']:.4f} | "
            f"top2 {metrics['top2_acc']:.4f} | "
            f"top3 {metrics['top3_acc']:.4f} | "
            f"sampled {metrics['sampled_acc']:.4f}"
        )

        # 以 top-1 准确率作为模型选择标准
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
            }, save_path)
            print(f"  → 保存最优模型 (top1={metrics['top1_acc']:.4f}, "
                  f"top2={metrics['top2_acc']:.4f}, top3={metrics['top3_acc']:.4f})")

    print(f"\n训练完成。最优 top-1 准确率: {best_val_acc:.4f}")
    print(f"模型已保存到: {save_path}")
    plot_losses(train_losses, val_losses, acc_curves, save_path="loss_curve_topk.png")


# ============================================================
# 8. 命令行入口
# ============================================================
def build_argparser():
    parser = argparse.ArgumentParser(description="Top-k 采样行为克隆训练脚本")
    parser.add_argument("--data", required=True, help=".npz 文件路径")
    parser.add_argument("--save", default="bc_model_topk.pt", help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")

    # Top-k 采样参数
    parser.add_argument("--top-k", type=int, default=3,
                        help="软标签和采样使用的 top-k 数量（默认 3，三张候选牌已足够）")
    parser.add_argument("--soft-alpha", type=float, default=0.15,
                        help="软标签中 top-k 分布的权重 α（0=纯硬标签，1=纯 top-k）")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="top-k 软标签的温度（越高越平滑）")
    parser.add_argument("--entropy-weight", type=float, default=0.01,
                        help="熵正则化权重（越大越鼓励分散）")
    return parser


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
