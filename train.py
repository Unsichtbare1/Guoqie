"""
麻将 AI 行为克隆训练脚本
用法:
    python train.py --data "output/huolongguo.npz" --save "huolongguo_model.pt"
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
def plot_losses(train_losses, val_losses, save_path="loss_curve.png"):
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
        plt.savefig(save_path, dpi=150)
        print(f"损失曲线已保存到: {save_path}")
    plt.show()


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
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
            total_correct += (pred == action).sum().item()
            total += action.size(0)
    return total_loss / total, total_correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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
        val_loss, val_acc = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")



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
    plot_losses(train_losses, val_losses, save_path="loss_curve.png")

# ============================================================
# 5. 命令行入口
# ============================================================
def build_argparser():
    parser = argparse.ArgumentParser(description="行为克隆训练脚本")
    parser.add_argument("--data", required=True, help=".npz 文件路径")
    parser.add_argument("--save", default="bc_model.pt", help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    return parser


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()


