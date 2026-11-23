"""
Phase 5 - The Hybrid GraphSAGE model.

Picture of one forward pass:

   node features ──► SAGEConv ─► SAGEConv ─► SAGEConv ──► node vectors h
                     (each: ReLU + dropout, hidden size 128)

   node vectors h ──► attention pool ┐
                   └─► mean pool ─────┤► graph embedding (256)
                                      │
   graph embedding ‖ 6 metrics ──► MLP(128→64→1) ──► one logit ──► sigmoid

"GraphSAGE" (SAGEConv) learns a node's vector by repeatedly mixing in its
neighbours' vectors. After 3 layers each node has "seen" neighbours up to 3 hops
away, so the vector captures local code structure.

Attention pooling: a tiny linear layer gives every node a score, we softmax those
scores within each graph, and take a weighted sum of the node vectors. The weights
say which nodes mattered most - we KEEP them for the line-level explanations in
Phase 6. We also concatenate a plain mean pool so no node is ever fully ignored.

Ablation switches (same class, different flags) let us answer "what helps?":
   - conv_type="gcn"     : use plain GCNConv instead of SAGEConv
   - use_metrics=False   : drop the 6 hand-crafted metrics (pure GNN)
   (AST-edges-only is handled in the data loader, not here.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax


class HybridGNN(nn.Module):
    def __init__(self, in_dim: int, metric_dim: int = 6, hidden: int = 128,
                 conv_type: str = "sage", use_metrics: bool = True,
                 dropout: float = 0.3, num_cwe: int = 0):
        super().__init__()
        self.use_metrics = use_metrics
        self.dropout = dropout
        self.num_cwe = num_cwe

        Conv = SAGEConv if conv_type == "sage" else GCNConv
        self.conv1 = Conv(in_dim, hidden)
        self.conv2 = Conv(hidden, hidden)
        self.conv3 = Conv(hidden, hidden)

        # attention pooling: one score per node
        self.att_lin = nn.Linear(hidden, 1)

        # graph embedding = [attention pool ‖ mean pool] = 2*hidden
        emb_dim = 2 * hidden + (metric_dim if use_metrics else 0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # STRETCH GOAL: optional second head that guesses the CWE type.
        # It reads the SAME graph embedding as the risk head ("multi-task
        # learning"). num_cwe=0 means no CWE head: the model is exactly the
        # original single-task Hybrid GraphSAGE.
        self.cwe_head = None
        if num_cwe > 0:
            self.cwe_head = nn.Sequential(
                nn.Linear(emb_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, num_cwe),
            )

    def embed(self, data):
        """Run the GNN + pooling and return (graph embedding, attention weights)."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv3(h, edge_index))

        # attention weights, normalised WITHIN each graph of the batch
        score = self.att_lin(h).squeeze(-1)          # [num_nodes]
        alpha = softmax(score, batch)                # [num_nodes], sums to 1 per graph
        att_pool = global_add_pool(h * alpha.unsqueeze(-1), batch)   # [B, hidden]
        mean_pool = global_mean_pool(h, batch)                        # [B, hidden]

        emb = torch.cat([att_pool, mean_pool], dim=1)
        if self.use_metrics:
            emb = torch.cat([emb, data.metrics], dim=1)
        return emb, alpha

    def forward(self, data, return_attention: bool = False, return_cwe: bool = False):
        """Risk logit per graph; optionally also attention weights and CWE logits.

        Returns `logit` alone by default (so older code keeps working), or a tuple
        (logit, [alpha], [cwe_logits]) in that order when asked for extras.
        """
        emb, alpha = self.embed(data)
        logit = self.mlp(emb).squeeze(-1)            # [B]

        extras = []
        if return_attention:
            extras.append(alpha)
        if return_cwe:
            if self.cwe_head is None:
                raise ValueError("this model was built without a CWE head (num_cwe=0)")
            extras.append(self.cwe_head(emb))        # [B, num_cwe]
        return (logit, *extras) if extras else logit
