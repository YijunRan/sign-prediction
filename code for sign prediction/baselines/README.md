# 符号链路预测基线模型

这是一个基于图神经网络的符号链路预测项目，专门用于预测带符号网络中边的符号（正负）。

## 项目结构

```
sign_prediction-baseline/
├── README.md                           # 项目说明文档
├── environment.yaml                    # Conda环境配置文件
├── code/                              # 源代码目录
│   ├── run_link_sign_prediction.py   # 主运行脚本（您需要运行的文件）
│   ├── run_link_sign_direction_tasks.py
│   ├── process_signed_datasets.py
│   └── sssnet.py
├── datasets/                          # 数据集目录
│   └── sign prediction/
│       ├── processed/                 # 预处理后的数据
│       │   ├── dataset_stats.json     # 数据集统计信息
│       │   ├── *_node_mapping.json    # 节点映射文件
│       │   ├── train_set/             # 训练集
│       │   │   ├── bitcoin_alpha/
│       │   │   ├── bitcoin_otc/
│       │   │   ├── slashdot/
│       │   │   └── wiki/
│       │   └── test_set/              # 测试集
│       │       ├── bitcoin_alpha/
│       │       ├── bitcoin_otc/
│       │       ├── slashdot/
│       │       └── wiki/
│       └── unified_processed/         # 统一处理后的数据
└── torch_geometric_signed_directed/   # 自定义PyTorch Geometric扩展
    ├── data/                          # 数据处理模块
    ├── nn/                            # 神经网络模型
    │   ├── signed/                    # 符号网络模型
    │   │   ├── SGCN.py               # 符号图卷积网络
    │   │   ├── SDGNN.py              # 符号有向图神经网络
    │   │   ├── SiGAT.py              # 符号图注意力网络
    │   │   └── SNEA.py               # 符号网络嵌入分析
    │   ├── directed/                  # 有向网络模型
    │   └── general/                   # 通用模型
    └── utils/                         # 工具函数
```

## 环境配置

### 1. 创建Conda环境

```bash
conda env create -f environment.yaml
```

### 2. 激活环境

```bash
conda activate sgnn
```

## 数据集

项目支持4个符号网络数据集：

- **bitcoin_alpha**: 比特币Alpha信任网络
- **bitcoin_otc**: 比特币OTC信任网络  
- **slashdot**: Slashdot社交网络
- **wiki**: 维基百科编辑者网络

每个数据集都包含正负边，用于符号链路预测任务。

## 支持的模型

项目实现了多种符号网络神经网络模型：

- **SGCN**: Signed Graph Convolutional Network
- **SDGNN**: Signed Directed Graph Neural Network
- **SiGAT**: Signed Graph Attention Network
- **SNEA**: Signed Network Embedding Analysis

## 运行代码

### 快速开始

运行符号链路预测的基本命令：

```bash
cd code
python run_link_sign_prediction.py --dataset bitcoin_alpha --model SDGNN
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `bitcoin_alpha` | 数据集名称 (bitcoin_alpha/bitcoin_otc/slashdot/wiki) |
| `--model` | str | `SDGNN` | 模型类型 (SGCN/SDGNN/SiGAT/SNEA) |
| `--epochs` | int | `500` | 训练轮数 |
| `--lr` | float | `0.01` | 学习率 |
| `--weight_decay` | float | `1e-3` | 权重衰减 |
| `--seed` | int | `0` | 随机种子 |
| `--in_dim` | int | `20` | 输入维度 |
| `--out_dim` | int | `20` | 输出维度 |
| `--eval_step` | int | `10` | 评估间隔 |
| `--patience` | int | `25` | 早停耐心值 |
| `--early_stop` | bool | `False` | 是否启用早停 |
| `--min_epochs` | int | `100` | 最小训练轮数 |
| `--split_id` | int | `0` | 数据分割ID |
| `--evaluate_all` | bool | `False` | 是否评估所有分割 |

### 运行示例

1. **使用默认参数运行**：
```bash
python run_link_sign_prediction.py
```

2. **指定数据集和模型**：
```bash
python run_link_sign_prediction.py --dataset wiki --model SiGAT
```

3. **启用早停机制**：
```bash
python run_link_sign_prediction.py --early_stop --patience 20
```

4. **评估所有数据分割**：
```bash
python run_link_sign_prediction.py --evaluate_all
```

5. **自定义训练参数**：
```bash
python run_link_sign_prediction.py --dataset slashdot --model SGCN --epochs 1000 --lr 0.005 --weight_decay 5e-4
```

## 输出结果

程序会输出以下评估指标：

- **Accuracy**: 准确率
- **Precision**: 精确率  
- **Recall**: 召回率
- **F1**: F1分数
- **AUC**: ROC曲线下面积


