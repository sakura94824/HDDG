# models/DGCNN_HilbertMultiGraph.py
"""
DGCNN with Multi-Graph Fusion (论文结构)

三图融合：
  - A_S: 静态距离矩阵（仅由脑电极坐标生成）
  - A_L: 功能连接矩阵（由 DE 特征计算通道相关性）
  - A_H: Hilbert 引导注意力矩阵（PBE 相位偏置注意力）

输入：融合特征 (B, C, 15) = concat(DE, amplitude, phase)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalized_laplacian(a: torch.Tensor, eps: float = 1e-5):
    """计算归一化拉普拉斯矩阵，支持 (C,C) 或 (B,C,C)"""
    d = a.sum(dim=-1)
    d_inv_sqrt = torch.pow(d + eps, -0.5)
    if a.dim() == 3:
        d_mat = torch.diag_embed(d_inv_sqrt)
        i = torch.eye(a.size(-1), device=a.device).unsqueeze(0).expand(a.size(0), -1, -1)
    else:
        d_mat = torch.diag_embed(d_inv_sqrt)
        i = torch.eye(a.size(0), device=a.device)
    return i - d_mat @ a @ d_mat


class GraphConv(nn.Module):
    """Chebyshev 图卷积"""
    def __init__(self, k: int, in_channels: int, out_channels: int):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_stack(self, x: torch.Tensor, lap: torch.Tensor):
        if self.k == 1:
            return x.unsqueeze(1)
        t0 = x
        t1 = torch.matmul(lap, x)
        ts = [t0, t1]
        for _ in range(2, self.k):
            t2 = 2 * torch.matmul(lap, ts[-1]) - ts[-2]
            ts.append(t2)
        return torch.stack(ts, dim=1)

    def forward(self, x: torch.Tensor, lap: torch.Tensor):
        cp = self.chebyshev_stack(x, lap)
        cp = cp.permute(0, 2, 3, 1).contiguous()
        cp = cp.view(x.size(0), x.size(1), -1)
        return cp @ self.weight


# ---------- Hilbert 编码器与 PBE 注意力 ----------
class AmplitudeEncoder(nn.Module):
    def __init__(self, num_freqs=5, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_freqs, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, amplitude):
        """amplitude: (B, C, F) -> E_A: (B, C, d)"""
        B, C, F = amplitude.shape
        x = amplitude.view(B * C, F)
        return self.mlp(x).view(B, C, -1)


class PhaseEncoder(nn.Module):
    def __init__(self, num_freqs=5, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, phase):
        """phase: (B, C, F) -> E_Phi: (B, C, d)"""
        B, C, F = phase.shape
        sin_p = torch.sin(phase)
        cos_p = torch.cos(phase)
        x = torch.cat([sin_p, cos_p], dim=-1).view(B * C, -1)
        return self.mlp(x).view(B, C, -1)


class PhaseBiasedAttention(nn.Module):
    """
    PBE: Phase-Biased Attention（对齐论文公式）
    S   = ẼA · ẼA^T / √dk          （振幅自注意力，反映激活强度关联）
    B   = ẼΦ · ẼΦ^T / √dk          （相位自注意力，反映时序协调关联）
    Ŝ  = S + β·B                   （β 可学习，控制相位偏置强度）
    A_H = Softmax(Ŝ)
    """
    def __init__(self, embed_dim=64, dropout=0.1):
        super().__init__()
        self.scale = embed_dim ** -0.5
        # 振幅投影
        self.W_A = nn.Linear(embed_dim, embed_dim, bias=False)
        # 相位投影
        self.W_Phi = nn.Linear(embed_dim, embed_dim, bias=False)
        # 可学习相位偏置系数 β
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, E_A, E_Phi):
        """
        E_A  : (B, C, d)  振幅嵌入
        E_Phi: (B, C, d)  相位嵌入
        Returns: A_H (B, C, C) 注意力邻接矩阵
        """
        # 振幅自注意力分数
        E_A_proj  = self.W_A(E_A)                                              # (B, C, d)
        S = torch.matmul(E_A_proj, E_A_proj.transpose(-1, -2)) * self.scale   # (B, C, C)

        # 相位自注意力偏置
        E_Phi_proj = self.W_Phi(E_Phi)                                         # (B, C, d)
        B = torch.matmul(E_Phi_proj, E_Phi_proj.transpose(-1, -2)) * self.scale  # (B, C, C)

        # 相位偏置注意力分数
        S_hat = S + self.beta * B                                              # (B, C, C)

        A_H = F.softmax(S_hat, dim=-1)
        A_H = self.dropout(A_H)
        A_H = 0.5 * (A_H + A_H.transpose(-1, -2))
        eye = torch.eye(A_H.size(1), device=A_H.device).unsqueeze(0)
        A_H = A_H * (1 - eye)
        return torch.clamp(A_H, min=0.0)


# ---------- DE 功能连接矩阵 A_L ----------
class FunctionalConnectivityModule(nn.Module):
    """
    从 DE 特征计算功能连接矩阵 A_L
    使用可学习的 Conv/Linear 提取通道表示，再计算相关性
    """
    def __init__(self, num_freqs=5, embed_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_freqs, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, de_features):
        """
        de_features: (B, C, F)
        Returns: A_L (B, C, C) 功能连接矩阵
        """
        B, C, n_f = de_features.shape
        h = self.encoder(de_features.view(B * C, n_f)).view(B, C, -1)
        Q = self.W_q(h)
        K = self.W_k(h)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        A_L = F.softmax(scores, dim=-1)
        A_L = self.dropout(A_L)
        A_L = 0.5 * (A_L + A_L.transpose(-1, -2))
        eye = torch.eye(C, device=A_L.device).unsqueeze(0)
        A_L = A_L * (1 - eye)
        return torch.clamp(A_L, min=0.0)


# ---------- 节点特征编码器 ----------
class NodeEncoder(nn.Module):
    """
    从 EEG 特征（DE + amplitude + phase 拼接）编码节点特征 X
    """
    def __init__(self, in_dim=15, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """x: (B, C, 15) -> X: (B, C, d)"""
        B, C, D = x.shape
        return self.mlp(x.view(B * C, D)).view(B, C, -1)


# ---------- 自适应多图融合 ----------
class AdaptiveTwoGraphFusion(nn.Module):
    """
    两图加权融合（用于消融）：A = softmax(w)_0 * G1 + softmax(w)_1 * G2
    G1, G2 形状均为 (B, C, C)
    """

    def __init__(self, init_weights=(0.5, 0.5)):
        super().__init__()
        logits = [np.log(w + 1e-8) for w in init_weights]
        self.fusion_logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, G1: torch.Tensor, G2: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.fusion_logits, dim=0)
        A = weights[0] * G1 + weights[1] * G2
        A = torch.clamp(A, min=0.0)
        A = 0.5 * (A + A.transpose(-1, -2))
        return A


class AdaptiveGraphFusion(nn.Module):
    """
    自适应融合 A_S, A_L, A_H
    A = w_s * A_S + w_l * A_L + w_h * A_H
    """
    def __init__(self, init_weights=(0.33, 0.33, 0.34), use_static=True):
        super().__init__()
        self.use_static = use_static
        logits = [np.log(w + 1e-8) for w in init_weights]
        self.fusion_logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, A_S, A_L, A_H):
        """
        A_S: (C, C) 或 (1, C, C) 或 None
        A_L, A_H: (B, C, C)
        Returns: A (B, C, C)
        """
        B = A_H.size(0)
        weights = F.softmax(self.fusion_logits, dim=0)
        if self.use_static and A_S is not None:
            w_s, w_l, w_h = weights[0], weights[1], weights[2]
            if A_S.dim() == 2:
                A_S = A_S.unsqueeze(0).expand(B, -1, -1)
            A = w_s * A_S + w_l * A_L + w_h * A_H
        else:
            w_l, w_h = weights[1], weights[2]
            w_sum = w_l + w_h + 1e-8
            A = (w_l * A_L + w_h * A_H) / w_sum
        A = torch.clamp(A, min=0.0)
        A = 0.5 * (A + A.transpose(-1, -2))
        return A


# ---------- 主模型 ----------
# 图消融模式（对应论文表格 tab:ablation_graph）
GRAPH_ABLATION_MODES = (
    "full",       # HDDG：15 维节点 + A_S + A_L + A_H
    "single_as",  # 15 维节点 + 仅 A_S（对照：单空间图）
    "wo_ah",      # w/o Hilbert：仅 DE 节点 + A_S + A_L（无 A_H，严格退化基线）
    "wo_ah_as",   # w/o Hilbert：仅 DE 节点 + 仅 A_S
    "wo_al",      # 去掉 A_L，保留 15 维节点 + A_S + A_H
    "wo_as",      # 去掉 A_S，保留 15 维节点 + A_L + A_H
)

# 严格 w/o Hilbert：节点仅 DE(5)，且不计算 / 融合 A_H
_STRICT_WO_HILBERT = frozenset({"wo_ah", "wo_ah_as"})


def uses_de_only_nodes(graph_ablation: str) -> bool:
    return str(graph_ablation or "full").strip().lower() in _STRICT_WO_HILBERT


def uses_A_H_in_fusion(graph_ablation: str) -> bool:
    return str(graph_ablation or "full").strip().lower() in ("full", "wo_al", "wo_as")


class DGCNN_HilbertMultiGraph(nn.Module):
    """
    多图融合 DGCNN

    输入: (B, C, 15) = concat(DE, amplitude, phase)

    graph_ablation:
      - "full": 15 维节点 + 三图 softmax 融合
      - "single_as": 15 维节点 + 仅 $A_S$
      - "wo_ah": 仅 DE 节点 + $A_S$+$A_L$（严格 w/o Hilbert）
      - "wo_ah_as": 仅 DE 节点 + 仅 $A_S$（严格 w/o Hilbert 空间基线）
      - "wo_al" / "wo_as": 15 维节点 + 两图融合
    """
    def __init__(
        self,
        num_electrodes: int = 62,
        num_freqs: int = 5,
        num_classes: int = 3,
        embed_dim: int = 64,
        k: int = 2,
        layers: list = None,
        dropout_rate: float = 0.5,
        static_adj: torch.Tensor = None,
        fusion_init: tuple = (0.33, 0.33, 0.34),
        graph_ablation: str = "full",
    ):
        super().__init__()
        if layers is None:
            layers = [64, 128]

        self.C = num_electrodes
        self.F = num_freqs
        self.d = embed_dim
        self.num_classes = num_classes
        self.k = k

        ga = str(graph_ablation or "full").strip().lower()
        if ga not in GRAPH_ABLATION_MODES:
            raise ValueError(
                f"graph_ablation={graph_ablation!r} 非法，须在 {GRAPH_ABLATION_MODES} 之一")
        self.graph_ablation = ga
        self.de_only_nodes = uses_de_only_nodes(ga)
        self.compute_hilbert_branch = ga not in _STRICT_WO_HILBERT

        if self.compute_hilbert_branch:
            self.amp_encoder = AmplitudeEncoder(num_freqs, embed_dim)
            self.phase_encoder = PhaseEncoder(num_freqs, embed_dim)
            self.pbe_attention = PhaseBiasedAttention(embed_dim, dropout_rate)
        else:
            self.amp_encoder = None
            self.phase_encoder = None
            self.pbe_attention = None

        # DE -> A_L
        self.func_conn = FunctionalConnectivityModule(num_freqs, embed_dim, dropout_rate)

        node_in_dim = num_freqs if self.de_only_nodes else num_freqs * 3
        self.node_encoder = NodeEncoder(in_dim=node_in_dim, embed_dim=embed_dim)

        # 静态邻接 A_S（仅电极坐标距离矩阵）
        if static_adj is not None:
            if not torch.is_tensor(static_adj):
                static_adj = torch.tensor(static_adj, dtype=torch.float32)
            adj = static_adj.clone().float()
            adj = 0.5 * (adj + adj.t())
            adj = adj * (1 - torch.eye(num_electrodes, device=adj.device))
            self.register_buffer("A_S", torch.clamp(adj, min=0.0))
        else:
            self.register_buffer("A_S", None)

        # 仅空间图必须与静态矩阵同时存在（否则无法构图）
        if self.graph_ablation in ("single_as", "wo_ah_as") and self.A_S is None:
            raise ValueError(f"graph_ablation={self.graph_ablation!r} 需要静态邻接 static_adj")

        self.has_static = self.A_S is not None

        # 多图融合
        tw = fusion_init[:2] if len(fusion_init) >= 2 else (0.5, 0.5)
        tw = (float(tw[0]), float(tw[1]))
        self.tw_init = tw
        self.graph_fusion = AdaptiveGraphFusion(
            init_weights=fusion_init,
            use_static=self.has_static
        )
        # 消融：任意两图的独立二项融合（与三图 softmax 并存，forward 按需选用）
        self.fusion_pair = AdaptiveTwoGraphFusion(init_weights=tw)

        # 图卷积层
        self.gconvs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_feats = embed_dim
        for out_feats in layers:
            self.gconvs.append(GraphConv(k, in_feats, out_feats))
            self.bns.append(nn.BatchNorm1d(num_electrodes))
            in_feats = out_feats

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        hidden = 256
        self.fc1 = nn.Linear(num_electrodes * layers[-1], hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _expand_as_batch(self, A_S: torch.Tensor, B: int) -> torch.Tensor:
        """(C,C) → (B,C,C)，无梯度复制"""
        if A_S.dim() == 2:
            return A_S.unsqueeze(0).expand(B, -1, -1).contiguous()
        return A_S

    def _fuse_adjacency(self, A_S, A_L, A_H, batch_size: int | None = None):
        """按 graph_ablation 得到用于 Laplacian 的 A（B,C,C），非负且对称"""
        B = batch_size if batch_size is not None else A_L.size(0)
        if self.graph_ablation in ("single_as", "wo_ah_as"):
            assert A_S is not None
            A = torch.clamp(self._expand_as_batch(A_S, B), min=0.0)
            A = 0.5 * (A + A.transpose(-1, -2))
            return A
        if self.graph_ablation == "wo_ah":
            # A_S + A_L
            if A_S is None:
                raise ValueError("wo_ah 需要静态邻接 A_S")
            return self.fusion_pair(
                self._expand_as_batch(A_S, A_L.size(0)), A_L)
        if self.graph_ablation == "wo_al":
            if A_S is None:
                raise ValueError("wo_al 需要静态邻接 A_S")
            return self.fusion_pair(
                self._expand_as_batch(A_S, A_L.size(0)), A_H)
        if self.graph_ablation == "wo_as":
            return self.fusion_pair(A_L, A_H)

        # full：沿用 AdaptiveGraphFusion（无 A_S 时构造函数已设 use_static=False）
        return self.graph_fusion(A_S, A_L, A_H)

    @torch.no_grad()
    def get_fusion_weights(self):
        if self.graph_ablation in ("single_as", "wo_ah_as"):
            return {"A_S": 1.0, "A_L": 0.0, "A_H": 0.0}
        if self.graph_ablation == "wo_ah":
            w = F.softmax(self.fusion_pair.fusion_logits, dim=0)
            return {"A_S": w[0].item(), "A_L": w[1].item(), "A_H": 0.0}
        if self.graph_ablation == "wo_al":
            w = F.softmax(self.fusion_pair.fusion_logits, dim=0)
            return {"A_S": w[0].item(), "A_L": 0.0, "A_H": w[1].item()}
        if self.graph_ablation == "wo_as":
            w = F.softmax(self.fusion_pair.fusion_logits, dim=0)
            return {"A_S": 0.0, "A_L": w[0].item(), "A_H": w[1].item()}
        if self.A_S is not None:
            w = F.softmax(self.graph_fusion.fusion_logits, dim=0)
            return {
                "A_S": w[0].item(),
                "A_L": w[1].item(),
                "A_H": w[2].item(),
            }
        return {"A_L": 0.5, "A_H": 0.5}

    def forward(self, x):
        """
        Args:
            x: (B, C, 15) = concat(DE, amplitude, phase)
        Returns:
            logits: (B, num_classes)
        """
        B, C, _ = x.shape
        de = x[:, :, :5]

        if self.compute_hilbert_branch:
            amp = x[:, :, 5:10]
            phase = x[:, :, 10:15]
            E_A = self.amp_encoder(amp)
            E_Phi = self.phase_encoder(phase)
            A_H = self.pbe_attention(E_A, E_Phi)
        else:
            A_H = None

        if self.graph_ablation == "wo_ah_as":
            A_L = None
        else:
            A_L = self.func_conn(de)

        node_x = de if self.de_only_nodes else x
        X = self.node_encoder(node_x)

        A = self._fuse_adjacency(self.A_S, A_L, A_H, batch_size=B)

        # 5. 图卷积
        lap = normalized_laplacian(A)
        h = X
        for gc, bn in zip(self.gconvs, self.bns):
            h = gc(h, lap)
            h = bn(h)
            h = self.relu(h)
            h = self.dropout(h)

        h = h.reshape(B, -1)
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


# =============================================================================
# 注意力消融（Ablation on attention strategy, tab:ablation_attention）
#
# 四种策略，均用于替换 DGCNN_HilbertMultiGraph 中的 pbe_attention 模块，
# 其余结构（A_L / A_S / GCN / 融合权重）完全相同，以便公平对比。
#
# attn_mode 取值：
#   "amp_attn"    – 仅振幅自注意力（无相位项）
#   "phase_attn"  – 仅相位自注意力（无振幅项）
#   "cross_attn"  – 标准 Q/K/V 交叉注意力（振幅→Q，相位→K/V）
#   "pba_attn"    – Phase-Biased Attention（本文方法，默认）
# =============================================================================

ATTN_ABLATION_MODES = ("amp_attn", "phase_attn", "cross_attn", "pba_attn")


class AmpSelfAttention(nn.Module):
    """
    Amp-Attn：仅振幅自注意力
      S   = W_A(E_A) · W_A(E_A)^T / √d
      A_H = Softmax(S)
    """
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.W_A = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, E_A: torch.Tensor, E_Phi: torch.Tensor) -> torch.Tensor:
        # E_Phi 忽略
        Q = self.W_A(E_A)
        S = torch.matmul(Q, Q.transpose(-1, -2)) * self.scale
        A_H = F.softmax(S, dim=-1)
        A_H = self.dropout(A_H)
        A_H = 0.5 * (A_H + A_H.transpose(-1, -2))
        eye = torch.eye(A_H.size(1), device=A_H.device).unsqueeze(0)
        return torch.clamp(A_H * (1 - eye), min=0.0)


class PhaseSelfAttention(nn.Module):
    """
    Phase-Attn：仅相位自注意力
      B   = W_Φ(E_Φ) · W_Φ(E_Φ)^T / √d
      A_H = Softmax(B)
    """
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.W_Phi = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, E_A: torch.Tensor, E_Phi: torch.Tensor) -> torch.Tensor:
        # E_A 忽略
        Q = self.W_Phi(E_Phi)
        B_mat = torch.matmul(Q, Q.transpose(-1, -2)) * self.scale
        A_H = F.softmax(B_mat, dim=-1)
        A_H = self.dropout(A_H)
        A_H = 0.5 * (A_H + A_H.transpose(-1, -2))
        eye = torch.eye(A_H.size(1), device=A_H.device).unsqueeze(0)
        return torch.clamp(A_H * (1 - eye), min=0.0)


class StandardCrossAttention(nn.Module):
    """
    Cross-Attn：标准 Q/K/V 交叉注意力
      Q = W_Q(E_A)，K = W_K(E_Φ)，V = W_V(E_Φ)
      O = Softmax(Q·K^T/√d)·V           shape: (B, C, d)
      A_H = Softmax(O·O^T/√d)          shape: (B, C, C)
    两步设计使输出仍为对称邻接矩阵，与其他策略接口一致。
    """
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, E_A: torch.Tensor, E_Phi: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(E_A)
        K = self.W_K(E_Phi)
        V = self.W_V(E_Phi)
        # 第一步：Q×K^T 得到 (B,C,C) 注意力权重
        attn = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        # 第二步：加权聚合 V，再投影为邻接
        O = torch.matmul(attn, V)                                           # (B,C,d)
        S = torch.matmul(O, O.transpose(-1, -2)) * self.scale               # (B,C,C)
        A_H = F.softmax(S, dim=-1)
        A_H = self.dropout(A_H)
        A_H = 0.5 * (A_H + A_H.transpose(-1, -2))
        eye = torch.eye(A_H.size(1), device=A_H.device).unsqueeze(0)
        return torch.clamp(A_H * (1 - eye), min=0.0)


# 策略工厂：attn_mode -> 注意力类
_ATTN_CLS = {
    "amp_attn":   AmpSelfAttention,
    "phase_attn": PhaseSelfAttention,
    "cross_attn": StandardCrossAttention,
    "pba_attn":   PhaseBiasedAttention,
}


class DGCNN_HilbertMultiGraph_AttnAblation(DGCNN_HilbertMultiGraph):
    """
    注意力策略消融模型（对应 tab:ablation_attention）。

    继承 DGCNN_HilbertMultiGraph，只替换 pbe_attention 模块，
    其余结构（NodeEncoder / A_L / A_S / GCN / 多图融合）完全不变。

    attn_mode 取值（见 ATTN_ABLATION_MODES）：
      "amp_attn"    – Amp-Attn（仅振幅自注意力）
      "phase_attn"  – Phase-Attn（仅相位自注意力）
      "cross_attn"  – Cross-Attn（标准 Q/K/V 交叉注意力）
      "pba_attn"    – PBA-Attn（本文方法，等同于基类）
    """

    def __init__(self, attn_mode: str = "pba_attn", **kwargs):
        attn_mode = attn_mode.lower().strip()
        if attn_mode not in ATTN_ABLATION_MODES:
            raise ValueError(
                f"attn_mode={attn_mode!r} 非法，须在 {ATTN_ABLATION_MODES} 之一"
            )
        super().__init__(**kwargs)
        self.attn_mode = attn_mode
        if attn_mode == "pba_attn":
            return
        embed_dim = kwargs.get("embed_dim", 64)
        dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.pbe_attention = _ATTN_CLS[attn_mode](embed_dim, dropout_rate)

    # forward 直接继承父类，无需重写
