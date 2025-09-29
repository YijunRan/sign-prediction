import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict

# 数据集配置
datasets = ["bitcoin_alpha", "bitcoin_otc", "slashdot", "wiki"]
splits = 5  # 每个数据集的分割数量

# 定义路径
base_dir = "../datasets/sign prediction"
train_dir = os.path.join(base_dir, "train_set")
test_dir = os.path.join(base_dir, "test_set")
output_dir = os.path.join(base_dir, "processed")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
for dataset in datasets:
    os.makedirs(os.path.join(output_dir, "train_set", dataset), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test_set", dataset), exist_ok=True)

# 进行节点映射的函数
def create_node_mapping(dataset_name):
    """为指定数据集创建节点映射"""
    all_nodes = set()
    
    # 收集所有文件中的所有节点
    for split_id in range(splits):
        # 处理训练集
        train_file = os.path.join(train_dir, dataset_name, f"train0.9_{split_id}.csv")
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file, names=["source", "target", "label"])
            all_nodes.update(train_df["source"].tolist())
            all_nodes.update(train_df["target"].tolist())
        
        # 处理测试集
        test_file = os.path.join(test_dir, dataset_name, f"test0.1_{split_id}.csv")
        if os.path.exists(test_file):
            test_df = pd.read_csv(test_file, names=["source", "target", "label"])
            all_nodes.update(test_df["source"].tolist())
            all_nodes.update(test_df["target"].tolist())
    
    # 创建映射: 原始节点ID -> 连续节点ID(从0开始)
    node_mapping = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    
    return node_mapping, len(node_mapping)

# 应用节点映射处理数据集
def process_dataset(dataset_name, node_mapping):
    """使用节点映射处理数据集的训练集和测试集"""
    print(f"处理数据集: {dataset_name}")
    
    dataset_stats = {
        "name": dataset_name,
        "splits": [],
        "node_count": len(node_mapping),
        "original_max_node_id": max(node_mapping.keys()),
        "new_max_node_id": max(node_mapping.values())
    }
    
    for split_id in range(splits):
        split_stats = {"id": split_id, "train": {}, "test": {}}
        
        # 处理训练集
        train_file = os.path.join(train_dir, dataset_name, f"train0.9_{split_id}.csv")
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file, names=["source", "target", "label"])
            
            # 应用节点映射
            train_df["source"] = train_df["source"].map(node_mapping)
            train_df["target"] = train_df["target"].map(node_mapping)
            
            # 统计信息
            split_stats["train"]["edge_count"] = len(train_df)
            split_stats["train"]["positive_edges"] = sum(train_df["label"] == 1)
            split_stats["train"]["negative_edges"] = sum(train_df["label"] == 0)
            
            # 保存处理后的文件
            output_train_file = os.path.join(output_dir, "train_set", dataset_name, f"train0.9_{split_id}.csv")
            train_df.to_csv(output_train_file, index=False, header=False)
            print(f"  保存训练集分割 {split_id} 到 {output_train_file}")
        
        # 处理测试集
        test_file = os.path.join(test_dir, dataset_name, f"test0.1_{split_id}.csv")
        if os.path.exists(test_file):
            test_df = pd.read_csv(test_file, names=["source", "target", "label"])
            
            # 应用节点映射
            test_df["source"] = test_df["source"].map(node_mapping)
            test_df["target"] = test_df["target"].map(node_mapping)
            
            # 统计信息
            split_stats["test"]["edge_count"] = len(test_df)
            split_stats["test"]["positive_edges"] = sum(test_df["label"] == 1)
            split_stats["test"]["negative_edges"] = sum(test_df["label"] == 0)
            
            # 保存处理后的文件
            output_test_file = os.path.join(output_dir, "test_set", dataset_name, f"test0.1_{split_id}.csv")
            test_df.to_csv(output_test_file, index=False, header=False)
            print(f"  保存测试集分割 {split_id} 到 {output_test_file}")
        
        dataset_stats["splits"].append(split_stats)
    
    return dataset_stats

# 处理所有数据集
all_stats = []
all_mappings = {}

for dataset in datasets:
    print(f"\n==================== 处理数据集: {dataset} ====================")
    # 创建节点映射
    node_mapping, node_count = create_node_mapping(dataset)
    print(f"找到 {node_count} 个唯一节点，原始最大节点ID: {max(node_mapping.keys())}, 新最大节点ID: {max(node_mapping.values())}")
    
    # 保存节点映射
    mapping_file = os.path.join(output_dir, f"{dataset}_node_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(node_mapping, f)
    print(f"保存节点映射到 {mapping_file}")
    
    # 存储节点映射以供后续使用
    all_mappings[dataset] = node_mapping
    
    # 处理数据集
    stats = process_dataset(dataset, node_mapping)
    all_stats.append(stats)

# 保存处理后的所有数据集统计信息
stats_file = os.path.join(output_dir, "dataset_stats.json")
with open(stats_file, 'w') as f:
    json.dump(all_stats, f, indent=2)
print(f"\n保存所有数据集统计信息到 {stats_file}")

# 打印汇总信息
print("\n==================== 处理完成 ====================")
for stats in all_stats:
    dataset = stats["name"]
    node_count = stats["node_count"]
    original_max = stats["original_max_node_id"]
    new_max = stats["new_max_node_id"]
    compression_ratio = original_max / (new_max + 1) if new_max > 0 else 0
    
    print(f"数据集 {dataset}:")
    print(f"  - 节点数: {node_count}")
    print(f"  - 原始最大节点ID: {original_max}")
    print(f"  - 新最大节点ID: {new_max}")
    print(f"  - 节点ID压缩比: {compression_ratio:.2f}x")
    
    for split in stats["splits"]:
        split_id = split["id"]
        if "train" in split and "edge_count" in split["train"]:
            train_edges = split["train"]["edge_count"]
            train_pos = split["train"]["positive_edges"]
            train_neg = split["train"]["negative_edges"]
            train_ratio = train_pos / train_edges if train_edges > 0 else 0
            
            print(f"  - 分割 {split_id} 训练集: {train_edges} 条边 (正: {train_pos}, 负: {train_neg}, 正比例: {train_ratio:.2f})")
        
        if "test" in split and "edge_count" in split["test"]:
            test_edges = split["test"]["edge_count"]
            test_pos = split["test"]["positive_edges"]
            test_neg = split["test"]["negative_edges"]
            test_ratio = test_pos / test_edges if test_edges > 0 else 0
            
            print(f"  - 分割 {split_id} 测试集: {test_edges} 条边 (正: {test_pos}, 负: {test_neg}, 正比例: {test_ratio:.2f})")

print("\n脚本执行完毕。处理后的数据集已保存到:", output_dir) 