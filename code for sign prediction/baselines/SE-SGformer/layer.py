#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.utils import degree, to_undirected, to_dense_adj

from diffusion import Diffusion

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class CentralityEncoding(nn.Module):
    def __init__(self, max_degree: int, node_dim: int):
        """
        :param max_degree: max pos degree or max neg degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_degree = max_degree
        self.node_dim = node_dim
        self.z_pos = nn.Parameter(torch.randn((max_degree, node_dim)))
        self.z_neg = nn.Parameter(torch.randn((max_degree, node_dim)))

    def forward(self, x: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.tensor) -> torch.Tensor:
        """
        :param x: node feature
        :param pos_edge_index: positive edge index
        :param neg_edge_index: positive edge index
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]
        # 浣跨敤 torch.bincount 璁＄畻姝ｅ害鏁板拰璐熷害鏁�
        positive_degrees = torch.bincount(pos_edge_index[0], minlength=num_nodes)
        negative_degrees = torch.bincount(neg_edge_index[0], minlength=num_nodes)

        # 浣跨敤 decrease_to_max_value 鍑芥暟闄愬埗搴︽暟鐨勬渶澶у€�
        positive_degrees = self.decrease_to_max_value(positive_degrees, self.max_degree - 1)
        negative_degrees = self.decrease_to_max_value(negative_degrees, self.max_degree - 1)

        # 灏嗘瘡涓妭鐐规搴﹀拰璐熷害鐨勬暟鍊间綔涓虹储寮曪紝鎸戦€墇_pos鎴杬_neg鐨勬瘡琛岋紝褰㈡垚姣忎釜鑺傜偣鐨勫祵鍏�
        x += self.z_pos[positive_degrees] + self.z_neg[negative_degrees]
        return x

    def decrease_to_max_value(self, x, max_value):
        "闄愬埗鑺傜偣搴︾殑鏈€澶у€�"
        x[x > max_value] = max_value
        return x


class RWEncoding(torch.nn.Module):
    def __init__(self, num):
        super().__init__()
        self.num = num  # 鍘熸潵鐨� num 淇濈暀

    def forward(self, feature: Tensor, num_nodes: int) -> Tensor:
        """
        feature: [num_walks, ???, ???]
        num_nodes: 褰撳墠 CSV 鑺傜偣鏁�
        """
        # 濡傛灉 feature 灏哄澶т簬 num_nodes锛屾埅鏂�
        if feature.shape[1] != num_nodes or feature.shape[2] != num_nodes:
            feature = feature[:, :num_nodes, :num_nodes]

        # 鐢熸垚 spatial_matrix
        spatial_matrix = torch.sum(feature, dim=0)  # [num_nodes, num_nodes]
        return spatial_matrix




class ADJEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_edge_index: Tensor, neg_edge_index: Tensor, num_nodes: torch.tensor) -> torch.Tensor:
        """
        :param pos_edge_index: positive edge index
        :param neg_edge_index: positive edge index
        :param num_nodes: number of nodes
        :return:  adjacency matrix encoding
        """
        # 确保num_nodes是整数
        num_nodes = int(num_nodes)
        
        # 创建邻接矩阵，大小为 num_nodes x num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        
        # 过滤掉超出范围的索引
        pos_mask = (pos_edge_index[0] < num_nodes) & (pos_edge_index[1] < num_nodes)
        neg_mask = (neg_edge_index[0] < num_nodes) & (neg_edge_index[1] < num_nodes)
        
        pos_edge_index_filtered = pos_edge_index[:, pos_mask]
        neg_edge_index_filtered = neg_edge_index[:, neg_mask]
        
        # 设置正边
        adj_matrix[pos_edge_index_filtered[0], pos_edge_index_filtered[1]] = 1
        adj_matrix[pos_edge_index_filtered[1], pos_edge_index_filtered[0]] = 1
        
        # 设置负边
        adj_matrix[neg_edge_index_filtered[0], neg_edge_index_filtered[1]] = -1
        adj_matrix[neg_edge_index_filtered[1], neg_edge_index_filtered[0]] = -1
        
        row_sum = adj_matrix.sum(dim=1, keepdim=True)
        epsilon = 1e-10
        normalized_adj_matrix = adj_matrix / (row_sum + epsilon)
        return normalized_adj_matrix



class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension

        """
        super().__init__()

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                adj_matrix: torch.Tensor,
                spatial_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param pos_edge_index: positive edge index
        :param neg_edge_index: positive edge index
        :param num_nodes: number of nodes
        :return: torch.Tensor, node embeddings after attention operation
        """
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        a = self.compute_a(key, query)
        # print("spatial_matrix:", np.shape(spatial_matrix))
        # print("adj_matrix:", np.shape(adj_matrix))
        # print("a:", np.shape(a))
        a = a + adj_matrix + spatial_matrix
        softmax = torch.softmax(a, dim=-1)
        x = softmax.mm(value)

        return x

    def compute_a(self, key, query):
        "Query-Key product(normalization)"
        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        return a


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                adj_matrix: torch.Tensor,
                spatial_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param adj_matrix: adjacency matrix encoding
        :param spatial_matrix: signed random walk encoding
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, adj_matrix, spatial_matrix) for attention_head in self.heads
                ], dim = -1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, num_heads):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param num_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=num_heads,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self,
                x: torch.Tensor,
                adj_matrix: torch.Tensor,
                spatial_matrix: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h鈥�(l) = MHA(LN(h(l鈭�1))) + h(l鈭�1)
        h(l) = FFN(LN(h鈥�(l))) + h鈥�(l)

        :param x: node feature matrix
        :param adj_matrix: adjacency matrix encoding
        :param spatial_matrix: signed random walk encoding
        :return: torch.Tensor, node embeddings after SE-SGformer layer operations
        """
        x_prime = self.attention(self.ln_1(x), adj_matrix, spatial_matrix) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new