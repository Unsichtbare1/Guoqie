"""
麻将 AI 行为克隆训练脚本
用法:
    python train/train.py --data "data/output/huolongguo.npz" --save "models/huolongguo_model.pt"
"""
from __future__ import annotations

import argparse
import math
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
# 1. 数据集
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
        # 用来做训练/验证集划分，避免同一局游戏的数据泄漏
        self.game_index = data["game_index"]
        self.file_index = data["file_index"]

        # 用 (file_index, game_index) 唯一标识一局游戏
        self.game_keys = list(zip(
            self.file_index.tolist(),
            self.game_index.tolist(),
        ))

        assert len(self.obs) == len(self.masks) == len(self.action)

    def __len__(self):
        return len(self.action)


    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]).float(),       # (1012, 34)
            torch.from_numpy(self.masks[idx]).bool(),       # (46,)
            torch.tensor(self.action[idx], dtype=torch.long),
        )


def split_by_game(dataset: MahjongDataset, val_ratio: float = 0.1, seed: int = 42):
    """
    按“局”划分训练/验证集，避免同一局游戏的数据同时出现在两边。
    """
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
# 2. 模型
# ============================================================
class MahjongNet(nn.Module):
    """
    输入: (B, 1012, 34)
    输出: (B, 46) 的动作 logits

    设计思路:
    - 1012 个特征平面当作“通道”，34 种牌当作“空间维度”
    - 先用 1x1 卷积把通道降到 128（减少参数量）
    - 再用两个 3x3 卷积提取局部模式
    - 最后全连接输出 46 维
    """
    def __init__(self, in_channels: int = 1012, num_actions: int = 46, hidden: int = 256):
        super().__init__()
        # 情况 1：1x1 卷积压缩数据
        self.proj = nn.Conv1d(in_channels, 128, kernel_size=1)
        # 情况 2：3x3 卷积 + padding=1（保持长度）
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
        # x: (B, 1012, 34)
        x = self.proj(x)                       # (B, 128, 34)
        x = F.relu(self.bn1(self.conv1(x)))    # (B, hidden, 34)
        x = F.relu(self.bn2(self.conv2(x)))    # (B, hidden, 34)
        return self.head(x)                    # (B, 46)


# ============================================================
# 3. 掩码交叉熵损失
# ============================================================
def masked_cross_entropy(logits: torch.Tensor, action: torch.Tensor, mask: torch.Tensor):
    """
    logits: (B, 46)
    action: (B,)
    mask:   (B, 46) bool，True 表示合法
    """
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg_inf)
    return F.cross_entropy(logits, action)


# ============================================================
# 4. 训练 / 验证
# ============================================================
def plot_losses(train_losses, val_losses, save_path="results/loss_curve.png"):
    """
    train_losses: list[float]，每个 epoch 的训练损失
    val_losses:   list[float]，每个 epoch 的验证损失
    save_path:    可选，保存图片路径；为 None 则只显示
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()


    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"损失曲线已保存到: {save_path}")
    plt.show()


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    total_n2 = 0
    correct_n2 = 0
    topk_hits = {2: 0, 3: 0}
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
    neg_inf = torch.finfo(torch.float32).min
    with torch.no_grad():
        for obs, mask, action in loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)
            logits = model(obs)
            # 必须先在同一份 logits 上掩码，再用掩码后的 logits 算损失和
            # argmax；否则非法动作（hora/riichi 等）的原始 logit 偏大，
            # 会导致准确率指标失真，模型选择也随之失效。
            logits = logits.masked_fill(~mask, neg_inf)
            loss = F.cross_entropy(logits, action)
            total_loss += loss.item() * action.size(0)

            pred = logits.argmax(dim=-1)
            correct = pred == action
            total_correct += correct.sum().item()

            n_legal = mask.sum(dim=1)
            for v in n_legal.tolist():
                n_hist[v] = n_hist.get(v, 0) + 1

            # 合法动作数 >= 2 的样本才包含真实选择，单独汇总 top1
            n2 = n_legal >= 2
            total_n2 += n2.sum().item()
            correct_n2 += correct[n2].sum().item()

            for k in topk_hits:
                hit = (logits.topk(k, dim=-1).indices == action.unsqueeze(-1)).any(dim=-1)
                topk_hits[k] += hit.sum().item()

            # NCE = 1 - CE/log(n)：只在合法集合上做 softmax（logsumexp 保证
            # 数值稳定），逐样本计算后取平均；n==1 的样本没有选择空间，
            # 其 CE 恒为 0，会虚高指标，故排除。
            lse = torch.logsumexp(logits, dim=-1)
            ce_i = lse - logits.gather(1, action.unsqueeze(1)).squeeze(1)
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
            conf_n2 = torch.softmax(logits, dim=-1).max(dim=-1).values[n2]
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
        "top1_acc": total_correct / total,
        "top1_acc_n2": correct_n2 / total_n2 if total_n2 else float("nan"),
        "top2_acc": topk_hits[2] / total,
        "top3_acc": topk_hits[3] / total,
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
        "conf_gap": mean_conf - (correct_n2 / total_n2 if total_n2 else float("nan")),
        "ece": ece,
        "n_legal_hist": n_hist,
    }


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} | 随机种子: {args.seed}")

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

    # 记录训练和验证损失，用于绘图
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for obs, mask, action in train_loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)

            logits = model(obs)
            loss = masked_cross_entropy(logits, action, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * action.size(0)
            total += action.size(0)

        scheduler.step()

        train_loss = total_loss / total
        metrics = evaluate(model, val_loader, device)
        val_loss = metrics["loss"]
        val_acc = metrics["top1_acc"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
              f"top1(n>=2) {metrics['top1_acc_n2']:.4f} | "
              f"top2 {metrics['top2_acc']:.4f} | "
              f"top3 {metrics['top3_acc']:.4f} | "
              f"NCE {metrics['nce']:.4f} | "
              f"PPL {metrics['ppl']:.3f} | "
              f"conf {metrics['mean_conf']:.4f} | gap {metrics['conf_gap']:+.4f} | "
              f"ECE {metrics['ece']:.4f}")
        buckets = metrics["ppl_buckets"]
        print("PPL 分桶: " + " | ".join(
            f"{name} PPL {buckets[name]['ppl']:.3f}"
            f" (n={buckets[name]['n']})"
            for name in ("2", "3-5", "6-10", ">=11")))
        if epoch == 1:
            hist = metrics["n_legal_hist"]
            print("验证集合法动作数分布: " + " ".join(
                f"{k}:{hist[k]}" for k in sorted(hist)))



        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "in_channels": in_channels,
                "num_actions": 46,
                "epoch": epoch,
                "val_acc": val_acc,
            }, save_path)
            print(f"  → 保存最优模型 (val_acc={val_acc:.4f})")

    print(f"\n训练完成。最优验证准确率: {best_val_acc:.4f}")
    print(f"模型已保存到: {save_path}")
    plot_losses(train_losses, val_losses, save_path=args.curve)

# ============================================================
# 5. 命令行入口
# ============================================================
def build_argparser():
    parser = argparse.ArgumentParser(description="行为克隆训练脚本")
    parser.add_argument("--data", required=True, help=".npz 文件路径")
    parser.add_argument("--save", default="models/bc_model.pt", help="模型保存路径")
    parser.add_argument("--curve", default="results/loss_curve.png", help="训练曲线保存路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="模型初始化与 DataLoader shuffle 的种子（数据划分固定 42）")
    return parser


def main():
    args = build_argparser().parse_args()
    train(args)



if __name__ == "__main__":
    main()

"""
    model = MahjongNet(in_channels=1012, num_actions=46, hidden=256)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")  # 4,906,670
"""


