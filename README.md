<h2 align="center"> Parasitic Capacitance and Risistance Predictor Based on GNN </h2>
** This is initial version of GNN4RC **


# DLPL-CAP: RC Circuit Capacitance and Resistance Prediction using GNNs

DLPL-CAP 是一个基于图神经网络 (Graph Neural Networks, GNN) 的深度学习项目，主要用于预测集成电路 (特别是 SRAM RC 电路) 中的寄生参数 (电容和电阻)。该项目将电路的网表拓扑和物理属性建模为图结构，能够在图层面上完成复杂的节点/边分类和回归任务。

## 🎯 项目目标
- **边缘任务 (分类/回归)**: 鉴别并预测边 (pair_to) 上的电阻值 (Resistance, $0 \sim 700 \Omega$)。
- **节点任务 (分类/回归)**: 预测网络节点 (net) 上的寄生电容值 (Capacitance, 约为 $10^{-13} F$ 级别)。

本项目在回归任务中采用创新的 **两阶段回归策略 (Two-stage Regression)**:
1. **分类器 (CapClassifier)**: 首先预测样本所在的值域区间（Class）。
2. **特定的回归器 (CapRegressor)**: 根据分配的分类区间，使用特定类别的多层感知机 (MLP) 进一步进行精确回归，极大地提升了异构、大动态范围电容/电阻预测的精确度。

---

## 📁 核心项目结构

```text
├── main.py               # 项目主入口，处理命令行参数、设置随机种子、初始化日志并拉起训练流。
├── model.py              # 核心 GNN 模型结构定义，包含 CapClassifier 和 CapRegressor。
├── sram_dataset.py       # 数据处理和加载逻辑，将 PyTorch Graph 转化为适用于 Gnn 训练的数据。
├── downstream_train.py   # 训练与评估引擎，包含损失计算、精确度评估、Focal Loss与AMP混合精度训练实现。
├── sampling.py           # 图数据采样工具 (子图采样，正负样本平衡)。
├── utils.py              # 项目实用工具 (模型参数保存/加载等)。
├── logs/                 # 记录各次训练输出的详细日志文件 (.txt)。
├── models_dlpl-cap-classifier/  # 自动保存的分类模型权重 (.pth)。
└── models_dlpl-cap-regressor/   # 自动保存的回归模型权重 (.pth)。
```

---

## 🛠️ 安装环境与依赖

本项目基于 **PyTorch** 和 **PyTorch Geometric (PyG)**。建议使用 `conda` 创建虚拟环境。

主要依赖:
- `python >= 3.8`
- `torch`
- `torch_geometric`
- `scikit-learn`
- `numpy`
- `tqdm`

---

## 🚀 训练与使用说明

通过运行 `main.py` 启动模型的训练与测试。您可以配置训练集、测试集、GPU、任务类型等超参数。

### 1. 分类任务 (Classification)
预测特定边或节点是否达到某个条件/是否存在对应的耦合效应。
```bash
python main.py \
    --task classification \
    --train_dataset "1+5+7+15+23+29" \
    --test_dataset "11+55+78" \
    --gpu 0 \
    --epochs 200 \
    --lr 0.00005 \
    --batch_size 64
```

### 2. 回归任务 (Regression)
精确预测 RC 电路中 net 的电容值或边的电阻值。
```bash
python main.py \
    --task regression \
    --train_dataset "1+5+7+15+23+29" \
    --test_dataset "11+55+78" \
    --gpu 0 \
    --epochs 200 \
    --lr 0.00005 \
    --batch_size 64 \
    --class_boundaries "0.33,0.67" 
```

### 主要命令行参数 (`main.py`)
- `--task`: `classification` 或 `regression`。
- `--train_dataset`: 训练数据集的 Case ID, 使用 `+` 号拼接多个。
- `--test_dataset`: 测试数据集的 Case ID。
- `--data_dir`: RC 数据集的目录（默认为 `../data`）。
- `--gpu`: 使用的GPU索引（如 `-1` 强制使用CPU，`0` 使用第一张卡）。
- `--epochs`: 总训练轮数（默认为 `200`）。
- `--num_gnn_layers`: GNN特征提取层的层数。
- `--hid_dim`: 模型隐含层特征维度（默认为 `32`）。
- `--use_amp`: 使用 AMP（自动混合精度）加速训练（默认为 `1`，开启）。
- `--use_focal_loss`: 对于严重不平衡的分类集是否启用 Focal Loss（默认为 `0`）。

---

## 🧠 核心模块说明

#### `model.py`
- `CapClassifier`: 输入电路图特征，经过节点属性编码 (Device, Pin, Net)、电路统计特征拼接，基于 `SAGEConv` (或其他 GNN 变体) 聚合各层表征来完成图/节点的分类。
- `CapRegressor`: 在回归任务中担任 "Stage-2"。包含多个参数独立的 Regression Head(MLP)，它会根据 `CapClassifier` 预测得到的分类边界 `class_idx` 将节点送入特定的回归 Head 解码，有效缩小回归的方差。

#### `sram_dataset.py` 
将原始异构 RC 电路数据清洗并转化成同构图。
- 采用归一化处理（电容归一化、电阻基于 `log1p` 处理以应对动态范围过大）。
- 支持子图分割，负样本生成算法 (negative edge sampling)，支持通过传入 `sample_rates` 对样本做欠采样处理。

#### `downstream_train.py`
囊括了完整的深度学习 Pipeline，训练分为两个主要流：
- `train_classification_epoch`: 带可选的 Focus Loss 和自动混合精度的常规分类器训练循环，通过 ROC-AUC、F1、Precision、Recall 记录并保存最优模型。
- `train_regress_epoch`: 使用两阶交替训练策略，一阶通过分类 CrossEntropy Loss 进行回归象限收敛，二阶通过特定类别的 MSELoss 收敛绝对电容/电阻值。评估包含 MAE、MSE、RMSE、R2 等各项指标。

## 📈 输出与日志记录
- **终端输出与日志存档**: 所有标准输出会被自动重定向到 `logs/` 目录下，并以 `[任务类型]_[训练集]_to_[测试集]_[时间戳].txt` 机制命名以防止覆盖。
- **最佳模型存档**: 训练时自动评估并在发现更优的验证结果 (ACC/AUC for 分类, MAE/R2 for 回归) 时，会自动保存模型至对应的 `models_dlpl-cap-*/` 文件夹中。
