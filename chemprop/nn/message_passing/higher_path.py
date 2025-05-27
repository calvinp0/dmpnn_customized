# import torch
# from torch import nn
# from typing import List
# from torch_scatter import scatter_add
from chemprop.data import BatchMolGraph
from chemprop.nn.message_passing.base import BondMessagePassing
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
import math

# def find_paths(adj: List[List[int]], max_len: int) -> List[List[int]]:
#     paths: List[List[int]] = []
#     def dfs(path):
#         if len(path) > 1:
#             paths.append(path.copy())
#         if len(path) == max_len + 1:
#             return
#         for nbr in adj[path[-1]]:
#             if nbr not in path:
#                 path.append(nbr); dfs(path); path.pop()
#     for start in range(len(adj)):
#         dfs([start])
#     return [p for p in paths if len(p) >= 3]  # keep ℓ=2,3

class PathMessagePassing(BondMessagePassing):
    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        undirected: bool = False,
        max_path_len: int = 3,
    ):
        super().__init__(
            d_v=d_v,
            d_e=d_e,
            d_h=d_h,
            bias=False,
            depth=depth,
            dropout=dropout,
            activation=activation,
            undirected=undirected,
        )
        pass
#         assert max_path_len in (2, 3)
#         self.max_path_len = max_path_len
#         # one linear for ℓ=2 and one for ℓ=3
#         self.path_linears = nn.ModuleList([
#             nn.Linear(2*d_e + 1*d_h, d_h, bias=False),
#             nn.Linear(3*d_e + 2*d_h, d_h, bias=False)
#         ][: (self.max_path_len - 1)])
#         for lin in self.path_linears:
#             nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))

#     def forward(self, bmg: BatchMolGraph, V_d=None) -> torch.Tensor:
#         # 1) standard BondMessagePassing to get H (E×d_h)
#         bmg = self.graph_transform(bmg)
#         H0 = self.initialize(bmg)
#         H  = self.tau(H0)
#         for _ in range(1, self.depth):
#             M_e = self.message(H, bmg)
#             H   = self.update(M_e, H0)

#         # 2) build adjacency & edge_map
#         ei = bmg.edge_index.cpu().tolist()
#         nV = bmg.V.size(0)
#         adj = [[] for _ in range(nV)]
#         edge_pairs = list(zip(ei[0], ei[1]))
#         edge_map   = {pair: idx for idx, pair in enumerate(edge_pairs)}
#         for u, v in edge_pairs:
#             adj[v].append(u)

#         idx = bmg.edge_index[1].unsqueeze(1).expand(-1, H.size(1))
#         M_nodes = torch.zeros(len(bmg.V), H.size(1), device=H.device) \
#                         .scatter_reduce_(0, idx, H, reduce="sum", include_self=False)
#         H_nodes = self.finalize(M_nodes, bmg.V, V_d)  # (V x d_h)

#         # 3) Enumerate & inject ℓ‐path messages including node states
#         for ℓ, linear in enumerate(self.path_linears, start=2):
#             raw_paths = find_paths(adj, max_len=ℓ)
#             paths     = [p for p in raw_paths if len(p) == ℓ+1]
#             print(f"ℓ={ℓ} → {len(paths)} paths (each of size {len(paths[0]) if paths else 0})")

#             if not paths:
#                 continue

#             feat_list, tgt = [], []
#             for p in paths:
#                 # collect ℓ edge feats
#                 e_ids = [
#                     edge_map.get((p[i], p[i+1]), edge_map[(p[i+1], p[i])])
#                     for i in range(len(p)-1)
#                 ]
#                 edge_feats = torch.cat([bmg.E[e] for e in e_ids], dim=-1)

#                 # collect ℓ−1 node states
#                 node_feats = torch.cat([H_nodes[p[i]] for i in range(1, len(p)-1)], dim=-1)

#                 # full path feature
#                 feat_list.append(torch.cat((edge_feats, node_feats), dim=-1))

#                 # scatter back onto the *last* edge
#                 tgt.append(e_ids[-1])

#             E_paths = torch.stack(feat_list, dim=0).to(H.device)
#             P       = linear(E_paths)                     # now linear must match new input dim
#             idx_e   = torch.tensor(tgt, device=H.device)
#             msg     = torch.zeros_like(H)
#             msg.scatter_add_(0, idx_e.unsqueeze(1).expand(-1, H.size(1)), P)
#             H = H + msg

#         # 4) aggregate back to nodes and finalize
#         idx = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
#         M   = torch.zeros(len(bmg.V), H.shape[1], device=H.device)\
#                    .scatter_reduce_(0, idx, H, reduce="sum", include_self=False)
#         return self.finalize(M, bmg.V, V_d)
