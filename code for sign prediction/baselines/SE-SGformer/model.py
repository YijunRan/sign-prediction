#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Union, Tuple, Dict, List, Optional
import scipy.sparse
import torch
import networkx as nx
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.utils import (
    coalesce,
    negative_sampling,
    structured_negative_sampling,
)

from layer import GraphormerEncoderLayer, CentralityEncoding, RWEncoding, ADJEncoding
from tmp import genWalk, generate_data
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class SE_SGformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_layers = args.num_layers
        self.num = args.num
        self.input_node_dim = args.num_node_features
        self.node_dim = args.node_dim
        self.output_dim = args.output_dim
        self.num_heads = args.num_heads
        self.max_degree = args.max_degree
        self.length = args.length
        self.max_hop = args.max_hop
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)

        self.centrality_encoding = CentralityEncoding(
            max_degree=self.max_degree,
            node_dim=self.node_dim
        )
        self.spatial_matrix = RWEncoding(
            num=self.num,
        )

        self.adj_matrix = ADJEncoding()

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                num_heads=self.num_heads,
                # max_path_distance=self.max_path_distance
            ) for _ in range(self.num_layers)
        ])
        self.lin = torch.nn.Linear(2 * args.node_dim, 3)
        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def split_edges(
        self,
        edge_index: Tensor,
        test_ratio: float = 0.2,
    ) -> Tuple[Tensor, Tensor]:
        r"""Splits the edges :obj:`edge_index` into train and test edges.

        Args:
            edge_index (LongTensor): The edge indices.
            test_ratio (float, optional): The ratio of test edges.
                (default: :obj:`0.2`)
        """
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

    def create_spectral_features(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Creates :obj:`in_channels` spectral node features based on
        positive and negative edges.

        Args:
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`pos_edge_index` and
                :attr:`neg_edge_index`. (default: :obj:`None`)
        """
        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, num_nodes=N)
        val = val - 1

        # Borrowed from:
        # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.input_node_dim, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    def forward(self, x: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> torch.Tensor:
        """
        :return: torch.Tensor, output node embeddings
        """
        file = './train/train_edges.csv'  # å¯ä»¥ä¿æŒï¼Œä½†æ³¨æ„ç”Ÿæˆæ•°æ®ç”¨ num_nodes
        feature_file = "./sign_random/your_dataset_spatial_feature.pt"

        num_nodes = x.shape[0]  # å½“å‰ CSV èŠ‚ç‚¹æ€»æ•°

        # å¦‚æžœ feature æ–‡ä»¶ä¸å­˜åœ¨ï¼Œç”Ÿæˆå¹¶ä¿å­˜
        if not os.path.isfile(feature_file):
            # ä½¿ç”¨ CSV æ•°æ®é›†ç”Ÿæˆéšæœºæ¸¸èµ°ç‰¹å¾
            train_edge_index, train_val, adj_unsigned, adj_signed, degree, offset = generate_data(file, num_nodes)
            feature = genWalk(train_edge_index, adj_signed, degree, offset, num_nodes,
                              self.length, self.max_hop, self.num)  # æ³¨æ„è¿™é‡Œçš„ num_nodes
            torch.save(feature, feature_file)

        # åŠ è½½ feature
        feature = torch.load(feature_file).to(device)

        # ç”Ÿæˆ spatial_matrix
        epsilon = 1e-10
        feature_reciprocal = torch.reciprocal(feature + epsilon)
        row_sum = feature.sum(dim=2, keepdim=True)
        normalized_feature = feature_reciprocal / row_sum

        # çº¿æ€§æ˜ å°„ + ä¸­å¿ƒæ€§ç¼–ç 
        x = self.node_in_lin(x)
        x = self.centrality_encoding(x, pos_edge_index, neg_edge_index)

        # ç¡®ä¿ spatial_matrix ä¸Ž x èŠ‚ç‚¹æ•°åŒ¹é…
        spatial_matrix = self.spatial_matrix(normalized_feature, num_nodes=num_nodes).to(device)
        adj_matrix = self.adj_matrix(pos_edge_index, neg_edge_index, num_nodes).to(device)
        # print(np.shape(spatial_matrix))
        # print(np.shape(adj_matrix))

        for layer in self.layers:
            x = layer(x, adj_matrix, spatial_matrix)

        x = self.node_out_lin(x)
        torch.save(x, f"./output/your_dataset_embedding.pt")
        return x

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
    ) -> Tensor:
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: Tensor) -> Tensor:
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the overall objective.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + 5 * (loss_1 + loss_2)

    def test(
            self,
            z: Tensor,
            pos_edge_index: Tensor,
            neg_edge_index: Tensor,
    ) -> Tuple[float, float, float]:
        """Evaluates node embeddings z on positive and negative test
        edges by computing AUC, F1 score, and Accuracy (ACC).

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]

        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat([
            pred.new_ones((pos_p.size(0))),
            pred.new_zeros((neg_p.size(0)))
        ])
        pred_np, y_np = pred.numpy(), y.numpy()

        auc = roc_auc_score(y_np, pred_np)
        f1 = f1_score(y_np, pred_np, average='binary') if pred_np.sum() > 0 else 0
        acc = accuracy_score(y_np, pred_np)

        return auc, f1, acc
