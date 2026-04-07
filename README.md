<h2 align="center"> Parasitic Capacitance and Risistance Predictor Based on GNN </h2>
** This is initial version of GNN4RC **

# DLPL-CAP: 基于图神经网络的电路寄生电容预测框架

**DLPL-CAP** (Deep Learning for Parasitic Extraction - Capacitance) 是一个基于图神经网络 (GNN) 的深度学习项目，专为评估和预测 SRAM（静态随机存取存储器）电路等复杂集成电路图中的寄生耦合电容（Parasitic Coupling Capacitance）而设计。

本项目将电路网表和物理拓扑抽象为图结构数据，支持对寄生边进行**边分类**（链路预测/判断量级）以及**边回归**（精准预测寄生电容的具体数值）的下游任务。为了解决电容值分布跨度极大的痛点，整个流程采用了**分类引导回归 (Classification-guided Regression)** 的创新型两阶段模型。

---

## 模型架构 (Model Architecture)

该项目在核心网络文件 `model.py` 中实现了级联的双阶段图神经网络模型，主要包括两个部分：

*   **CapClassifier (电容分类器)**:
    *   **混合节点编码器**: 通过 `Embedding` 处理离散的节点类型属性（如 `NET`, `DEV`, `PIN`），并使用带有可学习权重的线性映射层编码包含多维物理电路统计特征的属性。
    *   **图卷积模块**: 支持多类型图卷积算子（如 `SAGEConv`, `GCNConv`, `GATConv` 等），配合激活函数和批归一化 (BatchNorm) 抽取深层拓扑语义。
    *   **角色**: 在*分类任务*中用于二分类判断是否存在寄生电容；在*回归任务*中用于多分类预测该边对应的电容值“量级范围”（Class Boundaries），缩小回归头的数据方差。
*   **CapRegressor (电容回归器)**:
    *   **特定类别的分层回归设计**: 基于分类器给出的类别，激活专用的回归网络。其内部由多个独立的回归头组成。
    *   这极大缓解了由于距离带来的指数衰减的长尾数据分布问题，提高了数值拟合（MSE / R2）的精确度。

---

## 数据集与特征 (Dataset and Features)

本框架采用 `sram_dataset.py` 中定制化的 InMemoryDataset 来进行电路图数据的建模和处理。它具有高度的数据集拼接（如 `dataset1+dataset2`）和灵活采样能力。

*   **节点 (Nodes)**: 3 种基础电路节点类型：线网 `NET`、器件 `DEV` 和引脚 `PIN`，矩阵加载电路结构中的关键寄生统计指标。
*   **边 (Edges)**: 包含*结构化连接* (引脚到器件等) 和*寄生耦合边* (包含 Pin 与 Net 之间、Pin 之间、Net 之间的寄生耦合现象作为预测目标)。

---

## 环境依赖 (Requirements)

在运行本项目前，请确保 Python 环境中已安装以下主要包（建议使用 Conda 搭配虚拟环境）：

*   `python >= 3.8`
*   `torch >= 1.10.0`
*   `torch_geometric`
*   `scikit-learn`
*   `numpy`
*   `tqdm`

---

## 使用方法 (Usage)

项目提供 `main.py` 作为一键测试入口。使用强大的带参数重载调度：

**示例 1: 运行分类任务 (Classification)**
执行寄生边存在性判断/链路预测。可启用 Focal Loss 处理类间不平衡，并采样 10% 训练数据。
```bash
python main.py \
    --task classification \
    --train_dataset "sandwich+ultra8t" \
    --test_dataset "ssram+digtime+timing_ctrl+array_128_32_8t" \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.00005 \
    --num_gnn_layers 4 \
    --hid_dim 32 \
    --gpu 0 \
    --use_focal_loss 1 \
    --train_sample_rate 0.1 \
    --use_amp 1
```

**示例 2: 运行回归任务 (Regression)**
执行第二阶段串联耦合电容值预测任务（支持统计数据融合并将其转化为无向图）：
```bash
python main.py \
    --task regression \
    --train_dataset "sandwich+ultra8t" \
    --test_dataset "ssram+digtime+timing_ctrl+array_128_32_8t" \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.00005 \
    --gpu 0 \
    --src_dst_agg concat \
    --use_stats 1 \
    --to_undirected 1
```

---

## 项目目录结构 (Project Structure)

```text
DLPL-CAP/
├── main.py                     # 全局执行入口点，参数解析，定义种子及统一的记录系统
├── downstream_train.py         # 核心训练流水线（包含 Trainer、混合精度、评价指标聚合机制）
├── model.py                    # 模型定义文件：包含 CapClassifier 和二段式 CapRegressor
├── sram_dataset.py             # 数据集模块：数据图的加工与异质图归一化解析
├── sampling.py                 # 图的子集采样与 DataLoader 生成支持
├── utils.py                    # 功能性函数池：权重路径检查、自动读取保存、正负边生成工具逻辑
├── README.md                   # 本说明文档
├── MIGRATION_GUIDE.md          # 库升级及项目迁移支持文档
├── logs/                       # 训练日志缓存
├── models_dlpl-cap-classifier/ # 自动建构的预训练权重与检查点文件存放目录
└── models_dlpl-cap-regressor/  # CapRegressor 模型参数检查点保存目录
```
