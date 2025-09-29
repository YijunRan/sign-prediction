import argparse
import time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch_geometric.seed import seed_everything
from torch_geometric_signed_directed.nn.signed import SGCN, SDGNN, SiGAT, SNEA
from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='SDGNN')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--in_dim', type=int, default=20)
    parser.add_argument('--out_dim', type=int, default=20)
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--min_epochs', type=int, default=100)
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--evaluate_all', action='store_true')
    return parser.parse_args()


args = parameter_parser()
seed_everything(args.seed)

dataset_name = args.dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, splited_data, eval_flag='test'):
    model.eval()
    with torch.no_grad():
        z = model()
    embeddings = z.cpu().numpy()
    train_X = splited_data['train']['edges'].cpu().numpy()
    test_X = splited_data[eval_flag]['edges'].cpu().numpy()
    train_y = splited_data['train']['label'].cpu().numpy()
    test_y = splited_data[eval_flag]['label'].cpu().numpy()
    accuracy, precision, recall, f1, auc_score = link_sign_prediction_logistic_function(
        embeddings, train_X, train_y, test_X, test_y)
    return {
        'acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }


def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = model.loss()
    if torch.isnan(loss).any():
        print("警告: 遇到NaN损失值，使用预设的损失值替代")
        return 1.0
    loss.backward()
    optimizer.step()
    return loss.item()


split_ids = list(range(5)) if args.evaluate_all else [args.split_id]
print(f"\n使用数据集: {dataset_name}, 评估分割: {split_ids}")

res = []

for split_id in split_ids:
    train_file = f"../datasets/sign prediction/processed/train_set/{dataset_name}/train0.9_{split_id}.csv"
    test_file = f"../datasets/sign prediction/processed/test_set/{dataset_name}/test0.1_{split_id}.csv"

    print(f"\n------------- 处理分割 {split_id} -------------")
    print(f"训练集文件: {train_file}")
    print(f"测试集文件: {test_file}")

    # 读取 CSV
    column_names = ['source', 'target', 'label']
    train_df = pd.read_csv(train_file, header=None, names=column_names, dtype=int)
    test_df = pd.read_csv(test_file, header=None, names=column_names, dtype=int)

    train_edges = train_df[['source', 'target']].values
    train_labels = train_df['label'].values
    test_edges = test_df[['source', 'target']].values
    test_labels = test_df['label'].values

    # 划分验证集
    train_edges_final, val_edges, train_labels_final, val_labels = train_test_split(
        train_edges, train_labels, test_size=0.1, random_state=args.seed, stratify=train_labels
    )

    print(f"\n数据集统计信息:")
    print(f"训练集: {len(train_edges_final)} 条边, 正边数: {(train_labels_final == 1).sum()}, 负边数: {(train_labels_final == 0).sum()}")
    print(f"验证集: {len(val_edges)} 条边, 正边数: {(val_labels == 1).sum()}, 负边数: {(val_labels == 0).sum()}")
    print(f"测试集: {len(test_edges)} 条边, 正边数: {(np.array(test_labels) == 1).sum()}, 负边数: {(np.array(test_labels) == 0).sum()}")

    num_nodes = max(train_edges.max(), test_edges.max()) + 1
    print(f"总节点数: {num_nodes}")

    # 转换为 tensor
    train_edge_index = torch.tensor(train_edges_final, dtype=torch.long, device=device)
    train_edge_label01 = torch.tensor(train_labels_final, dtype=torch.long, device=device)
    val_edge_index = torch.tensor(val_edges, dtype=torch.long, device=device)
    val_edge_label = torch.tensor(val_labels, dtype=torch.long, device=device)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long, device=device)
    test_edge_label = torch.tensor(test_labels, dtype=torch.long, device=device)

    # 构造 edge_index_s (±1 符号)
    edge_index_s = torch.empty((train_edge_index.size(0), 3), dtype=torch.long, device=device)
    edge_index_s[:, :2] = train_edge_index
    sign_pm1 = (train_edge_label01.float() * 2 - 1).long()
    edge_index_s[:, 2] = sign_pm1

    pos_e = int((edge_index_s[:, 2] == 1).sum().item())
    neg_e = int((edge_index_s[:, 2] == -1).sum().item())
    print(f"喂给 {args.model} 的边: 正边 {pos_e}, 负边 {neg_e}")

    if pos_e == 0 or neg_e == 0:
        raise RuntimeError(f"[split {split_id}] {args.model}: 正/负边有空，无法训练")

    # 构建模型
    in_dim, out_dim = args.in_dim, args.out_dim
    if args.model == 'SGCN':
        model = SGCN(num_nodes, edge_index_s, in_dim, out_dim, layer_num=2, lamb=5).to(device)
    elif args.model == 'SNEA':
        model = SNEA(num_nodes, edge_index_s, in_dim, out_dim, layer_num=2, lamb=4).to(device)
    elif args.model == 'SDGNN':
        model = SDGNN(num_nodes, edge_index_s, in_dim, out_dim).to(device)
    elif args.model == 'SiGAT':
        model = SiGAT(num_nodes, edge_index_s, in_dim, out_dim).to(device)

        # --- 过滤掉空的 edge_list/agg ---
        non_empty_aggs = []
        non_empty_edges = []
        for edges, agg in zip(model.edge_lists, model.aggs):
            if edges.numel() > 0:
                non_empty_aggs.append(agg)
                non_empty_edges.append(edges)
        model.aggs = non_empty_aggs
        model.edge_lists = non_empty_edges

        # --- 修正 mlp_layer 的输入维度 ---
        new_in_dim = out_dim * (len(model.aggs) + 1)
        model.mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(new_in_dim, out_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(out_dim, out_dim)
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {'train_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}
    best_auc, test_info, patience = 0, {}, args.patience

    for epoch in range(args.epochs):
        t = time.time()
        loss = train(model, optimizer)
        history['train_loss'].append(loss)

        if (epoch + 1) % args.eval_step == 0:
            eval_info = evaluate(model, {
                'train': {'edges': train_edge_index, 'label': train_edge_label01},
                'val': {'edges': val_edge_index, 'label': val_edge_label},
                'test': {'edges': test_edge_index, 'label': test_edge_label}
            }, eval_flag='val')
            history['val_auc'].append(eval_info['auc'])
            history['val_acc'].append(eval_info['acc'])
            history['val_f1'].append(eval_info['f1'])
            t = time.time() - t

            print(f'Val Time: {t:.3f}s, Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Acc:{eval_info["acc"]:.4f}, Pre:{eval_info["precision"]:.4f}, '
                  f'Recall:{eval_info["recall"]:.4f}, F1:{eval_info["f1"]:.4f}, '
                  f'AUC:{eval_info["auc"]:.4f}')

            if eval_info['auc'] > best_auc:
                best_auc = eval_info['auc']
                test_info = evaluate(model, {
                    'train': {'edges': train_edge_index, 'label': train_edge_label01},
                    'val': {'edges': val_edge_index, 'label': val_edge_label},
                    'test': {'edges': test_edge_index, 'label': test_edge_label}
                }, eval_flag='test')
                test_info['epoch'], test_info['split_id'] = epoch, split_id
                patience = args.patience
                print(f'  发现更好的模型! 当前最佳AUC: {best_auc:.4f}')
            else:
                patience -= 1

            if args.early_stop and patience <= 0 and epoch >= args.min_epochs:
                print("早停触发")
                break

    print(f'\n===== 分割 {split_id} 测试结果 =====')
    print(f'最佳模型（轮次 {test_info["epoch"]:03d}）:')
    print(f'Accuracy: {test_info["acc"]:.4f}, Precision: {test_info["precision"]:.4f}, '
          f'Recall: {test_info["recall"]:.4f}, F1: {test_info["f1"]:.4f}, AUC: {test_info["auc"]:.4f}')
    res.append(test_info)

# 汇总结果
v = np.array([(i['acc'], i['precision'], i['recall'], i['f1'], i['auc']) for i in res])
print(f"\n====== 总体测试结果：{args.model} ======")
print(f"评估的分割数: {len(split_ids)}")
print(f"Accuracy: {v[:,0].mean():.4f} ({v[:,0].std():.4f}) ")
print(f"Precision: {v[:,1].mean():.4f} ({v[:,1].std():.4f}) ")
print(f"Recall: {v[:,2].mean():.4f} ({v[:,2].std():.4f}) ")
print(f"F1: {v[:,3].mean():.4f} ({v[:,3].std():.4f}) ")
print(f"AUC: {v[:,4].mean():.4f} ({v[:,4].std():.4f}) ")

if len(split_ids) > 1:
    print("\n每个分割的结果:")
    for r in res:
        print(f"分割 {r['split_id']}: Acc={r['acc']:.4f}, Pre={r['precision']:.4f}, "
              f"Recall={r['recall']:.4f}, F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
