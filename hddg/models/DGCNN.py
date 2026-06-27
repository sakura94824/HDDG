import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------- utils ----------
def normalized_laplacian(a: torch.Tensor, eps: float = 1e-5):
    # a: (C,C) 非负、对称
    d = a.sum(dim=1)                         # (C,)
    d_inv_sqrt = torch.pow(d + eps, -0.5)    # (C,)
    d_mat = torch.diag_embed(d_inv_sqrt)     # (C,C)
    i = torch.eye(a.size(0), device=a.device)
    return i - d_mat @ a @ d_mat

class B1ReLU(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.relu = nn.ReLU()
    def forward(self, x):                    # (B,C,F)
        return self.relu(x + self.bias)

class B2ReLU(nn.Module):
    def __init__(self, n_nodes: int, feat_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, n_nodes, feat_dim))
        self.relu = nn.ReLU()
    def forward(self, x):                    # (B,C,F)
        return self.relu(x + self.bias)

class GraphConv(nn.Module):
    """Chebyshev 图卷积，输入 X: (B, C, Fin)"""
    def __init__(self, k: int, in_channels: int, out_channels: int):
        super().__init__()
        assert k >= 1, "Chebyshev order k must be >= 1"
        self.k = k
        self.weight = nn.Parameter(torch.empty(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    # @torch.no_grad()
    def _stack_k1(self, x):
        return x.unsqueeze(1)  # (B,1,C,Fin)

    def chebyshev_stack(self, x: torch.Tensor, lap: torch.Tensor):
        # x: (B,C,Fin), lap: (C,C) -> (B,k,C,Fin)
        # 与 LibEER 对齐：T_0 = ones（全1），T_1 = lap @ x
        t_ones = torch.ones_like(x)
        if self.k == 1:
            return t_ones.unsqueeze(1)
        t1 = lap @ x
        if self.k == 2:
            return torch.stack([t_ones, t1], dim=1)
        # k > 2: [ones, x, lap@x, 2*lap*(lap@x)-x, ...]
        ts = [t_ones, x, t1]
        for _ in range(3, self.k):
            t2 = 2 * (lap @ ts[-1]) - ts[-2]
            ts.append(t2)
        return torch.stack(ts, dim=1)

    def forward(self, x: torch.Tensor, lap: torch.Tensor):
        cp = self.chebyshev_stack(x, lap)           # (B,k,C,Fin)
        cp = cp.permute(0, 2, 3, 1).contiguous()    # (B,C,Fin,k)
        cp = cp.view(x.size(0), x.size(1), -1)      # (B,C,Fin*k)
        out = cp @ self.weight                      # (B,C,Fout)
        return out

# ---------- model ----------
class DGCNN(nn.Module):
    """
    复现风格：
      - 输入 (B,T,C,F) 或 (B,C,F)
      - 纯静态邻接（graph.alpha=1.0；learnable_adj=False）
      - time_reduce='conv'（深度可分离 + GAP）
      - 两层 Chebyshev GCN
    """
    def __init__(
        self,
        num_electrodes: int = 62,
        in_channels: int = 5,
        num_classes: int = 3,
        k: int = 2,
        relu_is: int = 1,
        layers: list[int] = [64, 128],
        dropout_rate: float = 0.5,
        time_reduce: str = "conv",                 # 'mean' | 'conv' | 'attn'
        static_adj: torch.Tensor | None = None,   # (C,C)
        static_adj_alpha: float = 1.0,            # 
        learnable_adj: bool = False,              # 
    ):
        super().__init__()
        self.C = num_electrodes
        self.Fin = in_channels
        self.num_classes = num_classes
        self.k = int(k)
        self.relu_is = int(relu_is)
        self.layers = list(layers)
        self.dropout_rate = float(dropout_rate)
        self.time_reduce = time_reduce.lower()
        self.static_adj_alpha = float(static_adj_alpha)
        self.learnable_adj = bool(learnable_adj)

        # 时间压缩
        if self.time_reduce == "mean":
            self.Fenc = self.Fin
            self.time_encoder = None
        elif self.time_reduce == "conv":
            self.Fenc = self.Fin
            self.temporal_dw = nn.Conv1d(self.C * self.Fin, self.C * self.Fin, kernel_size=5, padding=2, groups=self.C * self.Fin)
            self.temporal_pw = nn.Conv1d(self.C * self.Fin, self.C * self.Fenc, kernel_size=1, groups=self.C)
            self.temporal_bn = nn.BatchNorm1d(self.C * self.Fenc)
            self.temporal_act = nn.ReLU()
        elif self.time_reduce == "attn":
            self.Fenc = self.Fin
            self.time_attn = nn.Sequential(
                nn.Linear(self.Fin, self.Fin),
                nn.Tanh(),
                nn.Linear(self.Fin, 1)
            )
        else:
            raise ValueError("time_reduce must be one of ['mean','conv','attn']")

        # 邻接（与 LibEER 对齐：xavier 初始化 + bias，relu 激活）
        if self.learnable_adj:
            self.adj_raw = nn.Parameter(torch.empty(self.C, self.C))
            nn.init.xavier_uniform_(self.adj_raw)
            self.adj_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("adj_raw", None)
            self.register_parameter("adj_bias", None)
        self.register_buffer("I", torch.eye(self.C))

        if static_adj is not None:
            if not torch.is_tensor(static_adj):
                static_adj = torch.tensor(static_adj, dtype=torch.float32)
            assert static_adj.shape == (self.C, self.C)
            self.register_buffer("A_static", static_adj.clone().float())
        else:
            self.A_static = None

        # 可学习的融合权重（当同时存在静态和可学习邻接矩阵时）
        if self.learnable_adj and self.A_static is not None:
            init_alpha = static_adj_alpha
            init_logits = torch.tensor([
                np.log(init_alpha + 1e-8),
                np.log(1.0 - init_alpha + 1e-8)
            ])
            self.adj_fusion_logits = nn.Parameter(init_logits)
        else:
            self.register_parameter("adj_fusion_logits", None)

        # 图卷积堆叠
        self.gconvs = nn.ModuleList()
        in_feats = self.Fenc
        for out_feats in self.layers:
            self.gconvs.append(GraphConv(self.k, in_feats, out_feats))
            in_feats = out_feats

        self.brelus = nn.ModuleList()
        if self.relu_is == 1:
            for ch in self.layers:
                self.brelus.append(B1ReLU(ch))
        else:
            for ch in self.layers:
                self.brelus.append(B2ReLU(self.C, ch))

        self.dropout = nn.Dropout(self.dropout_rate)

        # 分类头
        hidden = 256
        self.fc1 = nn.Linear(self.C * self.layers[-1], hidden)
        self.fc2 = nn.Linear(hidden, self.num_classes)
        nn.init.xavier_normal_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    
    @torch.no_grad()
    def _save_init_adj(self):
        if hasattr(self, "adj_raw") and self.adj_raw is not None:
            self.register_buffer("adj_init", self.adj_raw.clone())
            
    @torch.no_grad()
    def set_static_adj(self, static_adj: torch.Tensor | None):
        if static_adj is None:
            self.A_static = None
        else:
            if not torch.is_tensor(static_adj):
                static_adj = torch.tensor(static_adj, dtype=torch.float32)
            self.register_buffer("A_static", static_adj.clone().float())

    @torch.no_grad()
    def get_fusion_weights(self):
        """获取当前的融合权重（和为1）"""
        if self.adj_fusion_logits is not None:
            weights = torch.softmax(self.adj_fusion_logits, dim=0)
            return {
                "static_weight": weights[0].item(),
                "learnable_weight": weights[1].item()
            }
        return None

    def _build_adj(self):
        # 纯静态
        if (not self.learnable_adj) or (self.A_static is not None and self.static_adj_alpha >= 1.0 - 1e-8):
            if self.A_static is None:
                raise ValueError("learnable_adj=False 但未提供 A_static（静态邻接）。")
            a = self.A_static.clone()
            a = 0.5 * (a + a.t())
            a = torch.clamp(a * (1.0 - self.I), min=0.0)
            return a

        # 仅可学习（严格对齐 LibEER：relu(adj + bias)，不对称化，不去对角）
        if self.A_static is None or self.static_adj_alpha <= 1e-8:
            return torch.relu(self.adj_raw + self.adj_bias)

        # 融合（使用可学习权重，权重和为1）
        a_learn = F.softplus(self.adj_raw)
        a_learn = 0.5 * (a_learn + a_learn.t())
        a_learn = a_learn * (1.0 - self.I)

        fusion_weights = torch.softmax(self.adj_fusion_logits, dim=0)
        w_static = fusion_weights[0]
        w_learn = fusion_weights[1]

        a = w_static * self.A_static + w_learn * a_learn
        a = 0.5 * (a + a.t())
        a = torch.clamp(a, min=0.0)
        a = a * (1.0 - self.I)
        return a

    def _time_reduce(self, x: torch.Tensor):
        # x: (B,T,C,F) 或 (B,C,F) -> (B,C,Fenc)
        if x.dim() == 3:
            return x
        B, T, C, F = x.shape
        if self.time_reduce == "mean":
            return x.mean(dim=1)
        elif self.time_reduce == "conv":
            xt = x.permute(0, 2, 3, 1).contiguous().view(B, C*F, T)  # (B,C*F,T)
            y = self.temporal_dw(xt)
            y = self.temporal_pw(y)
            y = self.temporal_bn(y)
            y = self.temporal_act(y)
            y = y.mean(dim=-1)                                      # GAP
            return y.view(B, C, self.Fenc)
        else:  # attn
            xc = x.permute(0, 2, 1, 3).contiguous()                 # (B,C,T,F)
            scores = self.time_attn(xc)                              # (B,C,T,1)
            weights = torch.softmax(scores.squeeze(-1), dim=-1)      # (B,C,T)
            y = (xc * weights.unsqueeze(-1)).sum(dim=2)              # (B,C,F)
            return y

    def forward(self, x: torch.Tensor):
        x = self._time_reduce(x)                # (B,C,F)
        a = self._build_adj()                   # (C,C)
        lap = normalized_laplacian(a)
        h = x
        for gc, br in zip(self.gconvs, self.brelus):
            h = gc(h, lap)                      # (B,C,F')
            h = self.dropout(h)                 # LibEER 顺序：conv → dropout → b_relu
            h = br(h)
        h = h.reshape(h.size(0), -1)            # (B, C*F_last)
        h = self.dropout(h)                     # LibEER：dropout → fc1 → dropout → fc2（无 ReLU）
        h = self.fc1(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        
        return logits

# ---------- regularization utils ----------

class SparseL2Regularization(nn.Module):
    """
    对单个张量（例如 learnable 邻接矩阵）施加 L2 正则约束：
        loss_reg = λ * ||A||₂
    """
    def __init__(self, l2_lambda: float = 1e-4):
        super().__init__()
        self.l2_lambda = float(l2_lambda)

    def forward(self, x: torch.Tensor):
        if x is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        return self.l2_lambda * torch.norm(x, p=2)


class NewSparseL2Regularization(nn.Module):
    """
    对整个模型参数集合施加 L2 正则：
        loss_reg = λ * Σ ||W_i||₂
    （相当于 weight decay 的手动版本）
    """
    def __init__(self, l2_lambda: float = 1e-5):
        super().__init__()
        self.l2_lambda = float(l2_lambda)

    def forward(self, model: nn.Module):
        reg = torch.tensor(0., device=next(model.parameters()).device)
        for p in model.parameters():
            reg += torch.norm(p, p=2)
        return self.l2_lambda * reg
