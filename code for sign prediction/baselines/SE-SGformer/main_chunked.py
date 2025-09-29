#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, csv, tempfile
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


def load_edges_from_csv(file):
    df = pd.read_csv(file, header=None, sep=r'\s+|,', engine='python')
    src = df[0].values
    dst = df[1].values
    label = df[2].values
    pos_mask = label == 1
    neg_mask = label == -1
    pos_edge_index = torch.tensor(np.stack([src[pos_mask], dst[pos_mask]]), dtype=torch.long).to(device)
    neg_edge_index = torch.tensor(np.stack([src[neg_mask], dst[neg_mask]]), dtype=torch.long).to(device)
    num_nodes = max(df[0].max(), df[1].max()) + 1
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
    datasets = ["slashdot"]

    sample_size = 2000        # 每轮随机抽多少正/负边
    num_repeats = 50           # 同一个train/test重复多少次取平均
    num_epochs_per_repeat = 400  # 每次子图训练多少 epoch

    for dataset in datasets:
        train_dir = os.path.join(base_train, dataset)
        test_dir = os.path.join(base_test, dataset)
        train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".csv")])
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])

        for run_id, (train_file, test_file) in enumerate(zip(train_files, test_files)):
            print(f"\n=== Running {dataset} - {run_id} ===")

            # === 读原始训练集和测试集 csv ===
            df_train = pd.read_csv(os.path.join(train_dir, train_file), header=None, sep=r'\s+|,', engine='python')
            df_train.columns = ['src', 'dst', 'label']
            df_test = pd.read_csv(os.path.join(test_dir, test_file), header=None, sep=r'\s+|,', engine='python')
            df_test.columns = ['src', 'dst', 'label']

            all_test_acc, all_test_auc, all_test_f1 = [], [], []
            total_time_start = time.time()

            for repeat in range(num_repeats):
                print(f"--- Repeat {repeat+1}/{num_repeats} ---")

                # 初始化模型
                model = SE_SGformer(args).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

                # ---- 抽样正负边 ----
                pos_df = df_train[df_train['label'] == 1]
                neg_df = df_train[df_train['label'] == -1]
                pos_sample = pos_df.sample(min(sample_size, len(pos_df)))
                neg_sample = neg_df.sample(min(sample_size, len(neg_df)))
                df_sample = pd.concat([pos_sample, neg_sample])

                # ---- 重映射节点ID ----
                all_nodes = pd.concat([df_sample['src'], df_sample['dst']]).unique()
                mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
                df_sample['src'] = df_sample['src'].map(mapping)
                df_sample['dst'] = df_sample['dst'].map(mapping)

                # ---- 写入临时训练csv ----
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpf:
                    df_sample.to_csv(tmpf.name, header=False, index=False)
                    tmp_train_csv = tmpf.name

                # ---- 加载抽样训练数据 ----
                pos_edge_index, neg_edge_index, num_nodes = load_edges_from_csv(tmp_train_csv)
                os.remove(tmp_train_csv)

                # ---- 测试集也用同样映射 ----
                df_test_mapped = df_test.copy()
                df_test_mapped = df_test_mapped[df_test_mapped['src'].isin(mapping) & df_test_mapped['dst'].isin(mapping)]
                df_test_mapped['src'] = df_test_mapped['src'].map(mapping)
                df_test_mapped['dst'] = df_test_mapped['dst'].map(mapping)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpf:
                    df_test_mapped.to_csv(tmpf.name, header=False, index=False)
                    tmp_test_csv = tmpf.name

                test_pos_edge_index, test_neg_edge_index, _ = load_edges_from_csv(tmp_test_csv)
                os.remove(tmp_test_csv)

                # ---- 划分训练/验证 ----
                train_pos_edge_index, val_pos_edge_index = model.split_edges(pos_edge_index)
                train_neg_edge_index, val_neg_edge_index = model.split_edges(neg_edge_index)

                # ---- 构造节点特征 ----
                x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index, num_nodes).to(device)

                # ---- 在子图上训练 num_epochs_per_repeat ----
                for epoch in trange(num_epochs_per_repeat, desc=f"{dataset}-{run_id}-rep{repeat+1}"):
                    model.train()
                    optimizer.zero_grad()
                    z = model(x, train_pos_edge_index, train_neg_edge_index)
                    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
                    loss.backward()
                    optimizer.step()

                # ---- 训练结束后在映射过的测试集评估 ----
                test_acc, test_auc, test_f1 = model.test(model(x, test_pos_edge_index, test_neg_edge_index),
                                                         test_pos_edge_index, test_neg_edge_index)
                all_test_acc.append(test_acc)
                all_test_auc.append(test_auc)
                all_test_f1.append(test_f1)

            total_time = time.time() - total_time_start
            # 平均测试指标
            mean_acc = float(np.mean(all_test_acc))
            mean_auc = float(np.mean(all_test_auc))
            mean_f1 = float(np.mean(all_test_f1))

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset, run_id, 0.001, 5e-4, num_epochs_per_repeat,
                    loss.item(), "-", "-", "-",  # 验证指标不统计
                    mean_acc, mean_auc, mean_f1, total_time
                ])
            print(f"Finished {dataset}-{run_id}: "
                  f"Mean_Test_ACC={mean_acc:.4f}, Mean_Test_AUC={mean_auc:.4f}, Mean_Test_F1={mean_f1:.4f}, Time={total_time:.2f}s")
