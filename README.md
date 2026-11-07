# 果切 · 麻将行为克隆

用雀魂高段玩家的牌谱训练一个「模仿人类决策」的麻将模型。

整条流水线：下载牌谱 → 解析成 tenhou JSON → 转成 MJAI 事件流 → 用 Rust 后端把每个决策点编码成观察向量 → 训练行为克隆模型 → 评估。模型只做模仿（masked cross-entropy），不涉及强化学习或自我对弈。

当前数据集：499 场对局、75,189 个决策点，每个决策点包含 `obs (1012×34)`、46 维合法动作掩码和玩家实际动作。

## 工作流程

### 0. 抓取牌谱 UUID

从牌谱屋获取目标玩家的对局列表。需要先向牌谱屋管理员申请 token，填入 `scripts/get_uuids.py` 顶部的 `TOKEN` 常量。

```powershell
.\.venv\Scripts\python.exe scripts\get_uuids.py
```

输出 `uuids_<起>_<止>.txt`，每行一个 UUID（形如 `260831-xxxxxxxx-...`，不是分享链接里的编码串）。

### 1. 下载牌谱

```powershell
# 首次：打开浏览器，手动登录雀魂账号
node scripts/download_browser.mjs login

# 下单个牌谱
node scripts/download_browser.mjs 260831-58de3ade-784c-4129-9b47-0d843cded5bc

# 按 UUID 列表批量下载（默认下载后自动转换成 JSON）
node scripts/download_browser.mjs --uuids-file uuids_20260101_20260831.txt
```

输入：UUID。输出：`data/paipu_raw/<uuid>.head.bin` 与 `.data.bin`（原始 protobuf 字节）。

雀魂接入阿里云 AWSC / 网易易盾风控，纯 WebSocket 直连登录会返回 `code=151`，所以下载只能走真实浏览器：脚本通过 Chrome DevTools Protocol 拦截客户端与服务器的 WebSocket 通信，把牌谱原始字节截下来。账号有下载配额，实测老号约 100 个/轮、新号约 10 个。

### 2. 原始数据转 tenhou JSON

```powershell
.\.venv\Scripts\python.exe scripts\convert_paipu.py data\paipu_raw data\paipu_json
```

输入：上一步的 `.bin`。输出：`data/paipu_json/paipu-<uuid>.json`（tenhou.net/6 通用格式）。解析依赖 `vendor/` 下 vendored 的雀魂 protobuf 定义与 tensoul 解析器。

### 3. tenhou JSON 转 MJAI

```powershell
.\.venv\Scripts\python.exe scripts\tenhou2mjai.py data\paipu_json data\dataset\mjai
```

输入：tenhou JSON。输出：`data/dataset/mjai/*.json`，逐行 JSONL 事件流（`start_kyoku` / `tsumo` / `dahai` / `chi` / `pon` / `reach` / `hora` / `ryukyoku` 等）。Mortal 的 Rust 后端只认这种格式。

### 4. 导出决策点

```powershell
.\.venv\Scripts\python.exe Mortal-main\mortal\export_decision_points.py `
    data\dataset\mjai `
    -o data\output\huolongguo `
    --version 4 `
    --player-names "むほうむてん"
```

输入：MJAI 目录 + 目标玩家名。输出三个文件：

- `data/output/huolongguo.npz` — 训练数据（`obs` / `masks` / `action` 及牌局元信息）
- `data/output/huolongguo.jsonl` — 每个决策点的可读元信息
- `data/output/huolongguo.json` — manifest（源文件、动作名、跳过文件）

玩家名会做 NFKC + 片假名→平假名 + casefold 归一化后宽松匹配，只加载包含目标玩家的牌谱。导出阶段会剔除「合法动作数 == 1」的样本（立直后强制摸切、单候选杠等没有真实选择空间），保证 npz 与 jsonl 的索引一一对应。当前数据中 `n_legal` 最小为 2、最大 16、平均 8.76。

### 5. 训练

两条线，共用同一份 npz 和同一套按局划分（`(file_index, game_index)` 为键，默认 9:1，避免同一局跨训练/验证集）。

MLP 基线：

```powershell
.\.venv\Scripts\python.exe train\train.py `
    --data data\output\huolongguo.npz `
    --save models\huolongguo_model_filtered.pt `
    --epochs 5 --batch-size 64 --lr 3e-4 --seed 42 `
    --curve results\loss_curve_mlp_5ep_filtered.png
```

Transformer（逐牌位读出）：

```powershell
.\.venv\Scripts\python.exe train\train-transformer.py `
    --data data\output\huolongguo.npz `
    --save models\huolongguo_transformer_per_tile_filtered.pt `
    --readout per_tile --epochs 20 --batch-size 64 --lr 3e-4 `
    --d-model 384 --nhead 8 --num-layers 4 --dim-ff 1024 --dropout 0.1 `
    --warmup-steps 400 --seed 42 --early-stop-patience 4 `
    --curve results\loss_curve_transformer_per_tile_filtered.png `
    --log results\train_transformer_per_tile_filtered.log
```

| 模型 | 结构 | 参数量 |
|---|---|---|
| `train/train.py` | 1×1 卷积压通道 → 两层 3×3 卷积 + BN → Flatten → 512 → 46 | 4.9 M |
| `train/train-transformer.py` | 34 个牌位 token → 4 层 8 头 Encoder → 逐牌位读出（37 类切牌直接对应牌位，9 类非切牌用 mean pooling） | 5.9 M |

关键开关：`--seed` 固定 torch/numpy；`--curve` 保存损失曲线；Transformer 的 `--early-stop-patience N` 表示连续 N 轮 top1 不升就停（0 关闭）；`--log` 同时把日志写入文件。MLP 按验证 top1 存最优权重，5 轮就饱和，不建议再加大轮数。

产物统一用「模型名 + 配置后缀」命名：`models/*.pt` 是权重，`results/*.log` 是日志，`results/loss_curve_*.png` 是曲线，三者同名，便于对照。

## 工具一览

| 脚本 | 作用 | 什么时候用 |
|---|---|---|
| `scripts/get_uuids.py` | 从牌谱屋抓目标玩家的对局 UUID | 数据准备的第一步 |
| `scripts/download_browser.mjs` | 浏览器 CDP 拦截 WebSocket，下载牌谱原始字节，默认下载后自动转换 | 唯一下载入口；需先 `login` |
| `scripts/convert_paipu.py` | protobuf → tenhou.net/6 JSON | 通常由下载脚本自动调用，也可手动 |
| `scripts/tenhou2mjai.py` | tenhou JSON → MJAI 事件流 | 导出决策点之前 |
| `scripts/json2jsongz.py` | 把 JSON 压成 `.json.gz` | 需要压缩存档时的小工具 |
| `Mortal-main/mortal/export_decision_points.py` | 用 `libriichi` 把 MJAI 编码成 obs / mask / action | 生成训练数据 |
| `train/train.py` | MLP 行为克隆训练 | 快速基线、验证数据质量 |
| `train/train-transformer.py` | Transformer 训练，`--readout per_tile` 为当前主路径 | 正式训练 |

## 环境与配置

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy torch matplotlib
npm ci
```

- Python：numpy / torch / matplotlib
- Node.js 18+：仅下载步骤需要（`protobufjs`、`ws`，由 `package.json` 声明）
- Rust 后端 `libriichi.pyd` 已预编译放在 `.venv/Lib/site-packages/`，负责解析 MJAI、生成 1012×34 观察向量和合法动作掩码，无需自行编译

下载行为由项目根目录的 `config.json` 控制，可从 `config.example.json` 复制修改；命令行参数优先级更高：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `raw_dir` | `data/paipu_raw` | 原始 protobuf 输出目录 |
| `json_dir` | `data/paipu_json` | tenhou JSON 输出目录 |
| `mjai_dir` | `data/dataset/mjai` | MJAI 目录（仅用于打印后续命令） |
| `browser_path` | `null` | Chrome/Edge 路径，留空自动查找 |
| `port` | `null` | CDP 调试端口，留空自动选择 |
| `timeout_sec` | `120` | 单局等待超时 |
| `keep_open` | `false` | 结束后是否保留浏览器窗口 |

数据落位约定：`data/paipu_raw`（原始字节）→ `data/paipu_json`（tenhou JSON）→ `data/dataset/mjai`（事件流）→ `data/output`（npz/jsonl）；权重写 `models/`，日志与曲线写 `results/`。数据、权重、日志都不入库。

## 指标

动作空间共 46 维：0–8 / 9–17 / 18–26 分别是打 1–9 万、筒、索，27–33 打字牌，34–36 打赤五，37 立直，38–40 吃（低/中/高），41 碰，42 杠，43 和了，44 流局，45 跳过。暗杠、加杠由后端自动处理，不在动作空间内。

- **top-k 准确率**：掩码后取概率最高的 k 个动作，看是否包含实际动作。
- **NCE** = `1 - CE / log(n)`：按合法动作数 n 归一化，逐样本平均。1 表示完美，0 表示在合法动作上均匀乱猜，负值比乱猜还差。它消除了「候选多导致 CE 天然大」的影响，适合跨局面比较。
- **PPL** = `exp(mean CE)`：困惑度，可理解为模型平均在几个候选之间犹豫，越低越好。另按 `n_legal` 分 2 / 3–5 / 6–10 / ≥11 四桶统计，用来定位多候选局面的难度。
- **置信度与 ECE**：合法动作上的最大 softmax 概率为置信度；ECE（15 个等宽区间）衡量置信度与实际准确率的偏差，越小说明校准越好。

验证集（7,614 条，seed 42）实测：

| 指标 | MLP | Transformer per_tile |
|---|---|---|
| top1 | 0.7096 | **0.7150** |
| top2 / top3 | 0.8981 / 0.9494 | 0.8966 / 0.9510 |
| NCE | 0.5646 | **0.5847** |
| PPL | 2.263 | 2.20 |
| 置信度 / ECE | 0.766 / 0.057 | 0.747 / **0.035** |

Transformer 的 PPL 分桶为 `2 → 1.427`、`3–5 → 1.724`、`6–10 → 2.469`、`≥11 → 2.529`；候选越多越难，这是当前主要瓶颈之一。

## 已知限制

- 行为克隆只模仿目标玩家的动作分布，没有奖励信号，不会探索新策略。
- 数据量（499 场）相对模型规模偏小，增加对局数比加宽模型更有效；下载受雀魂风控配额限制。
- 9 类非切牌动作（碰/吃/立直/和了等）与 37 类切牌共享一套 logits，容易尺度失衡，判断准确率明显低于切牌。
- top1 被「打牌 / 跳过」这类高频动作主导，比较模型时建议同时看 NCE 与 PPL。
