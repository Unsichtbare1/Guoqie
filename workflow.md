# 雀魂牌谱 → 行为克隆训练 全工作流

本仓库实现了从雀魂下载牌谱、转换格式、提取决策点、到训练模仿学习模型的完整流水线。

## 目录

- [整体流程](#整体流程)
- [环境准备](#环境准备)
- [第 0 步：获取uuids](#第-0-步获取uuids)
- [第 1 步：下载雀魂牌谱](#第-1-步下载雀魂牌谱)
- [第 2 步：原始数据转 tenhou JSON](#第-2-步原始数据转-tenhou-json)
- [第 3 步：tenhou JSON 转 MJAI](#第-3-步tenhou-json-转-mjai)
- [第 4 步：提取决策点](#第-4-步提取决策点)
- [第 5 步：训练模型](#第-5-步训练模型)
- [目录结构](#目录结构)


---

## 整体流程

```
雀魂服务器
  │
  │  ① download_browser.mjs（浏览器 CDP 拦截 WebSocket）
  ▼
paipu_raw/*.head.bin + *.data.bin（原始 protobuf 字节）
  │
  │  ② convert_paipu.py（vendored tensoul 解析器）
  ▼
paipu_json/paipu-<uuid>.json（tenhou.net/6 格式 JSON）
  │
  │  ③ tenhou2mjai.py（手牌/副露/摸打 流重构）
  ▼
dataset/mjai/*.json（MJAI 事件流，逐行 JSONL）
  │
  │  ④ export_decision_points.py（Rust libriichi GameplayLoader）
  ▼
output/huolongguo.npz（obs + masks + action 决策点数据集）
  │
  │  ⑤ train.py（PyTorch 行为克隆训练）
  ▼
huolongguo_model.pt（训练好的模型权重）
```

---

## 环境准备

### Python 环境

```powershell
# 在项目根目录创建虚拟环境（已存在则跳过）
python -m venv .venv

# 激活
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install numpy torch matplotlib
# libriichi 已预装在 .venv/Lib/site-packages/，无需额外安装
```

### Node.js 环境（仅下载步骤需要）

```powershell
# 需 Node.js 18+，安装后在项目根目录执行
npm install
# 安装 protobufjs 和 ws 依赖
```

### Rust libriichi（已预编译，无需操作）

#需要安装Mortal库。此方法中的提取决策点和生成观察向量的功能依赖于 Rust 编译的 `libriichi` 库。
`libriichi.pyd` 已放置在 `.venv/Lib/site-packages/` 中，是 Mortal 项目的 Rust 后端。它负责：
- 解析 MJAI 事件流，校验规则合法性
- 将牌谱状态编码为 1012×34 的观察向量
- 生成 46 维合法动作掩码

---
## 第 0 步：获取uuids

**脚本**：`get_uuids.py`

**意图**：从牌谱屋抓取目标玩家的牌谱 UUID 列表，供下载脚本使用。
注意，需要向管理员说明意图并申请token。


## 第 1 步：下载雀魂牌谱

**脚本**：`download_browser.mjs`

**意图**：通过本地浏览器（Edge/Chrome）的 Chrome DevTools Protocol 拦截雀魂客户端与服务器之间的 WebSocket 通信，截获牌谱的原始 protobuf 字节。

**为什么不能直接 API 登录下载**：雀魂接入了阿里云 AWSC / 网易易盾风控，纯 WebSocket 裸连无法通过认证（返回 code=151）。必须在真实浏览器环境中让风控指纹自然生成。

### 首次使用：账号密码登录

```powershell
node download_browser.mjs login
```

会打开浏览器，手动输入账号密码登录。
注：实测下来，我本人的账号能下载100个牌谱左右，新建的小号似乎只能下载10个牌谱。就本模型的训练中，我抓取大半年的一千多把牌谱只能下载到三百多个。
目前暂不知道限制频率为多久。

### 下载牌谱

```powershell
# 下载单个牌谱
node download_browser.mjs 260831-58de3ade-784c-4129-9b47-0d843cded5bc

# 批量下载（UUID 列表文件，每行一个 UUID）
node download_browser.mjs --uuids-file uuids.txt

# 下载后自动调用 convert_paipu.py 转换（默认行为）
# 加 --no-convert 可跳过转换
node download_browser.mjs 260831-58de3ade-... --no-convert
```

**输出**：
- `paipu_raw/<uuid>.head.bin` — 牌谱头部 protobuf（玩家信息、规则等）
- `paipu_raw/<uuid>.data.bin` — 牌谱体 protobuf（每局每步操作记录）

**注意**：UUID 是雀魂内部 ID（形如 `260831-xxxxxxxx-xxxx-...`），不是分享链接 `?paipu=` 后面的编码串。UUID 可从浏览器开发者工具的 Network 面板中抓取，或用 `download_paipu.mjs --probe` 探测。

---

## 第 2 步：原始数据转 tenhou JSON

**脚本**：`convert_paipu.py`

**意图**：将下载的原始 protobuf 字节解码为 tenhou.net/6 格式的 JSON。此格式是天凤/雀魂牌谱的通用交换格式，后续转换器都基于此格式工作。

**依赖**：`vendor/` 目录下的 vendored 组件：
- `vendor/ms/` — 雀魂 protobuf 协议定义（`protocol_pb2.py`）
- `vendor/tensoul/` — tensoul 牌谱解析器（`parser.py`、`model.py`），将 protobuf 记录重构为逐局 dump 格式

### 用法

```powershell
# 通常由 download_browser.mjs 自动调用，也可手动运行
.\.venv\Scripts\python.exe convert_paipu.py paipu_raw paipu_json
```

**输出**：`paipu_json/paipu-<uuid>.json`，每个文件包含完整一场对局（通常 8-12 局）。

**JSON 结构**（关键字段）：
```json
{
  "ver": "2.3",
  "name": ["玩家0", "玩家1", "玩家2", "玩家3"],
  "sc": [座位0分, 座位0点, ...],
  "rule": {"disp": "南", "aka53": 1, "aka52": 2, "aka51": 1},
  "log": [
    [局meta, scores, doras, uras, P0配牌/P0摸/P0打, P1..., P2..., P3..., 结果],
    ...
  ]
}
```

---

## 第 3 步：tenhou JSON 转 MJAI

**脚本**：`tenhou2mjai.py`

**意图**：将 tenhou 的「每玩家独立 draws/discards 数组」布局重构为 MJAI 的「时序事件流」格式。MJAI 是 Mortal Rust 后端能直接解析的格式（逐行 JSONL）。

### 用法

```powershell
.\.venv\Scripts\python.exe tenhou2mjai.py paipu_json dataset/mjai
```

**输出**：`dataset/mjai/paipu-<uuid>.json`，逐行 JSONL 事件流。

**MJAI 事件类型**：
- `start_game` / `end_game` — 场次起止
- `start_kyoku` — 局开始（配牌、宝牌、座位）
- `tsumo` — 摸牌
- `dahai` — 打牌
- `chi` / `pon` / `daiminkan` / `kakan` / `ankan` — 副露
- `reach` / `reach_accepted` — 立直
- `hora` — 和了
- `ryukyoku` — 流局
- `end_kyoku` — 局结束

---

## 第 4 步：提取决策点

**脚本**：`Mortal-main/mortal/export_decision_points.py`

**意图**：用 Rust 后端 `libriichi.dataset.GameplayLoader` 遍历 MJAI 事件流，在目标玩家的每个需要决策的时刻（摸牌后打牌、别人打牌后是否鸣牌等），将当前完整牌局状态编码为 1012×34 的观察向量，并记录该玩家实际采取的动作和当前合法动作掩码。

### 玩家名匹配

脚本会预扫描所有牌谱的 `start_game` 事件，提取四个玩家名，用 NFKC + 片假名→平假名 + casefold 归一化后与 `--player-names` 参数做宽松匹配。只有包含目标玩家的牌谱才会被加载。

### 用法

```powershell
.\.venv\Scripts\python.exe Mortal-main\mortal\export_decision_points.py `
    dataset/mjai `
    -o output/huolongguo `
    --version 4 `
    --player-names "むほうむてん" #果圣的id
```

**参数说明**：
- `dataset/mjai` — MJAI 牌谱目录
- `-o output/huolongguo` — 输出文件前缀（生成 .npz / .json / .jsonl）
- `--version 4` — 观察向量编码版本（v4 是最新版，1012 通道）
- `--player-names` — 目标玩家名（需要与牌谱内名字匹配）

**输出文件**：
- `output/huolongguo.npz` — 训练数据（numpy 压缩格式）
- `output/huolongguo.json` — manifest（源文件列表、动作名、跳过文件等）
- `output/huolongguo.jsonl` — 每个决策点的元信息（file_index、game_index、action_name、legal_action_names 等）

### npz 数据格式

| 字段 | shape | dtype | 说明 |
|------|-------|-------|------|
| `obs` | (N, 1012, 34) | float32 | 观察向量（1012 个特征平面 × 34 种牌） |
| `masks` | (N, 46) | bool | 合法动作掩码 |
| `action` | (N,) | int64 | 玩家实际采取的动作 ID |
| `player_id` | (N,) | int64 | 玩家座位（0-3） |
| `file_index` | (N,) | int64 | 源牌谱文件索引 |
| `game_index` | (N,) | int64 | 场内局序号 |
| `sample_index` | (N,) | int64 | 局内决策点序号 |
| `at_kyoku` | (N,) | int64 | 所在局号 |
| `at_turn` | (N,) | int64 | 所在巡目 |
| `shanten` | (N,) | int64 | 向听数 |
| `done` | (N,) | bool | 是否本局最后一个决策 |
| `apply_gamma` | (N,) | bool | RL 奖励折扣标记 |

### 46 个动作

| ID 范围 | 含义 |
|---------|------|
| 0-8 | 打 1m-9m |
| 9-17 | 打 1p-9p |
| 18-26 | 打 1s-9s |
| 27-33 | 打 东/南/西/北/白/发/中 |
| 34-36 | 打赤五（5mr/5pr/5sr） |
| 37 | 立直 |
| 38-40 | 吃低/吃中/吃高 |
| 41 | 碰 |
| 42 | 杠 |
| 43 | 和了 |
| 44 | 流局 |
| 45 | 跳过（不做鸣牌反应） |

---

## 第 5 步：训练模型

**脚本**：`train.py`

**意图**：使用掩码交叉熵损失对模型进行行为克隆训练。模型学习模仿目标玩家在各个决策点的动作选择。

### 模型架构

`MahjongNet`（定义在 `train.py` 中）：
- 输入：`(B, 1012, 34)` 的观察向量
- 1×1 卷积压缩通道到 128
- 两层 3×3 卷积（128→256）+ BatchNorm + ReLU
- 全连接层：256×34 → 512 → 46
- 输出：46 维 logits（对应 46 个动作）

### 用法

```powershell
# 基本训练（默认 5 epochs）
.\.venv\Scripts\python.exe train.py --data output/huolongguo.npz --save huolongguo_model.pt

# 调整超参数
.\.venv\Scripts\python.exe train.py `
    --data output/huolongguo.npz `
    --save huolongguo_model.pt `
    --epochs 20 `
    --batch-size 128 `
    --lr 3e-4 `
    --val-ratio 0.1
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | （必填） | .npz 文件路径 |
| `--save` | `bc_model.pt` | 模型保存路径 |
| `--epochs` | 5 | 训练轮数 |
| `--batch-size` | 64 | 批大小 |
| `--lr` | 3e-4 | 学习率 |
| `--val-ratio` | 0.1 | 验证集比例（按局划分，避免数据泄漏） |

### 训练策略

- **按局划分**：使用 `(file_index, game_index)` 唯一标识一局游戏，确保同一局不会同时出现在训练集和验证集中
- **掩码损失**：`masked_cross_entropy` 先将 logits 中非法动作位置设为 `-inf`，再计算交叉熵，确保模型只学习合法动作的概率分布
- **学习率调度**：余弦退火（CosineAnnealingLR）
- **梯度裁剪**：`max_norm=1.0`，防止梯度爆炸
- **最优模型保存**：按验证准确率保存最优模型

### 验证指标

训练集和验证集都会计算**掩码后的 argmax 准确率**（在合法动作中选概率最大的，看是否与实际动作一致）。当前数据集（38627 个决策点）3 epoch 后验证准确率约 69%。

---

## 目录结构

```
d:\Project-list\guoqie\
│
├── download_browser.mjs       # ① 浏览器下载牌谱
├── convert_paipu.py            # ② protobuf → tenhou JSON
├── tenhou2mjai.py              # ③ tenhou JSON → MJAI
├── train.py                    # ⑤ 行为克隆训练
│
├── vendor/                     # ② 的依赖
│   ├── ms/                     #   雀魂 protobuf 定义
│   └── tensoul/                #   tensoul 解析器
│
├── package.json                # ① 的 Node.js 依赖声明
├── config.example.json         #    下载配置示例
├── uuids.example.txt           #    UUID 列表示例
│
├── Mortal-main/
│   └── mortal/
│       └── export_decision_points.py  # ④ 提取决策点
│
├── .venv/                      # Python 虚拟环境（含 libriichi.pyd）
│
├── paipu_raw/                  # ① 输出：原始 protobuf 字节
├── paipu_json/                 # ② 输出：tenhou JSON
├── dataset/mjai/               # ③ 输出：MJAI 事件流
├── output/                     # ④⑤ 输出：npz 数据集 + 模型
│   ├── huolongguo.npz
│   ├── huolongguo.json
│   └── huolongguo.jsonl
├── get_uuids.py                 # 辅助脚本：抓取目标玩家的牌谱 UUID 列表
│
├── huolongguo_model.pt         # ⑤ 输出：训练好的模型
├── loss_curve.png              #    训练损失曲线
│
├── README.md                   # 本文档
└── 模型使用指南.md              # 模型加载与推理文档
```



## 完整一键流程

```powershell
# 0. 激活环境
.\.venv\Scripts\Activate.ps1
npm install  # 首次

# 1. 下载牌谱（自动转换）
node download_browser.mjs --uuids-file uuids.txt

# 2. 转换为 MJAI
.\.venv\Scripts\python.exe tenhou2mjai.py paipu_json dataset/mjai

# 3. 提取决策点
.\.venv\Scripts\python.exe Mortal-main\mortal\export_decision_points.py `
    dataset/mjai -o output/huolongguo --version 4 --player-names "むほうむてん"

# 4. 训练
.\.venv\Scripts\python.exe train.py `
    --data output/huolongguo.npz --save huolongguo_model.pt --epochs 5
```
