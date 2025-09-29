#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, csv
import pandas as pd
import numpy as np
import torch
from tqdm import trange
from model import SE_SGformer
from parameter import parse_args

# 选择设备
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device:", device)

def remap_ids(df_list):
    """
    接收若干个df（训练集、测试集等），返回映射后的df以及映射关系
    """
    all_nodes = pd.concat([df[0] for df in df_list] + [df[1] for df in df_list]).unique()
    mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
    new_dfs = []
    for df in df_list:
        df = df.copy()
        df[0] = df[0].map(mapping)
        df[1] = df[1].map(mapping)
        new_dfs.append(df)
    return new_dfs, mapping

def load_edges_from_df(df):
    """从已经映射过的df得到边"""
    src = df[0].values
    dst = df[1].values
    label = df[2].values
    pos_mask = label == 1
    neg_mask = label == -1
    
    # 确保索引从0开始连续
    all_nodes = np.unique(np.concatenate([src, dst]))
    num_nodes = len(all_nodes)
    
    # 重新映射索引以确保连续性
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
    src_mapped = np.array([node_mapping[node] for node in src])
    dst_mapped = np.array([node_mapping[node] for node in dst])
    
    pos_edge_index = torch.tensor(np.stack([src_mapped[pos_mask], dst_mapped[pos_mask]]), dtype=torch.long).to(device)
    neg_edge_index = torch.tensor(np.stack([src_mapped[neg_mask], dst_mapped[neg_mask]]), dtype=torch.long).to(device)
    
    return pos_edge_index, neg_edge_index, num_nodes

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(1147)

    results_file = "results.csv"
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Dataset", "Run", "LR", "WeightDecay", "Epochs",
                "Final_Loss", "Val_ACC", "Val_AUC", "Val_F1",
                "Test_ACC", "Test_AUC", "Test_F1", "Time(s)"
            ])

    base_train = "./sign prediction/train_set"
    base_test = "./sign prediction/test_set"
    datasets = ["bitcoinalpha", "bitcoinotc", "slashdot", "wiki"]

    for dataset in datasets:
        train_dir = os.path.join(base_train, dataset)
        test_dir = os.path.join(base_test, dataset)
        train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".csv")])
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])

        for run_id, (train_file, test_file) in enumerate(zip(train_files, test_files)):
            # 如果出现了报错显示RuntimeError: The size of tensor a (1980) must match the size of tensor b (1962) at
            # non-singleton dimension 1 ,请将sign_random下的特征文件清除，然后使用以下代码跳过之前的数据集编号从上次程序断掉的
            # 地方跑。
            # if run_id == 0 or run_id == 1:
            #     continue
            print(f"\n=== Running {dataset} - {run_id} ===")

            # === 读原始csv ===
            df_train = pd.read_csv(os.path.join(train_dir, train_file), header=None, sep=r'\s+|,', engine='python')
            df_test = pd.read_csv(os.path.join(test_dir, test_file), header=None, sep=r'\s+|,', engine='python')

            # === 压缩节点ID ===
            (df_train_mapped, df_test_mapped), mapping = remap_ids([df_train, df_test])
            print(f"节点数由原始 {max(df_train[0].max(), df_train[1].max())+1} 压缩到 {max(df_train_mapped[0].max(), df_train_mapped[1].max())+1}")

            # === 加载边 ===
            pos_edge_index, neg_edge_index, num_nodes = load_edges_from_df(df_train_mapped)

            # 初始化模型和优化器
            model = SE_SGformer(args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

            # 拆分训练/验证
            train_pos_edge_index, val_pos_edge_index = model.split_edges(pos_edge_index)
            train_neg_edge_index, val_neg_edge_index = model.split_edges(neg_edge_index)

            # 节点特征
            x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index, num_nodes).to(device)

            num_epochs = 400
            # sample_size = 1000  # 每轮随机抽样边的数目
            total_time_start = time.time()

            val_accs, val_aucs, val_f1s = [], [], []
            # 不抽样，直接用全部训练边
            for epoch in trange(num_epochs, desc=f"{dataset}-{run_id}"):
                model.train()
                optimizer.zero_grad()
                z = model(x, train_pos_edge_index, train_neg_edge_index)
                loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
                loss.backward()
                optimizer.step()

                acc, auc, f1 = model.test(
                    model(x, val_pos_edge_index, val_neg_edge_index),
                    val_pos_edge_index, val_neg_edge_index)
                val_accs.append(acc); val_aucs.append(auc); val_f1s.append(f1)


            total_time = time.time() - total_time_start

            # 平均验证指标
            val_acc = float(np.mean(val_accs[-10:]))
            val_auc = float(np.mean(val_aucs[-10:]))
            val_f1 = float(np.mean(val_f1s[-10:]))

            # 测试集（映射过）
            test_pos_edge_index, test_neg_edge_index, _ = load_edges_from_df(df_test_mapped)
            test_acc, test_auc, test_f1 = model.test(model(x, test_pos_edge_index, test_neg_edge_index),
                                                     test_pos_edge_index, test_neg_edge_index)

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset, run_id, 0.001, 5e-4, num_epochs,
                    loss.item(), val_acc, val_auc, val_f1,
                    test_acc, test_auc, test_f1, total_time
                ])
            print(f"Finished {dataset}-{run_id}: "
                  f"Val_ACC={val_acc:.4f}, Test_ACC={test_acc:.4f}, Time={total_time:.2f}s")
