import numpy as np
import pandas as pd

def _pcc_from_matrix(mat, eps=1e-8, absval=True, threshold=None):
    """
    由矩阵 mat (C, L) 计算 PCC (C, C)，数值稳定实现。
    """
    mat = np.asarray(mat, dtype=np.float64, order='C')
    C, L = mat.shape
    if L <= 1:
        raise ValueError("PCC: 每个通道的样本点太少 (L <= 1)。")

    # 去均值
    mean = mat.mean(axis=1, keepdims=True)        # (C,1)
    Xc = mat - mean                               # (C,L)

    # 协方差 (无偏)
    cov = (Xc @ Xc.T) / (L - 1)                   # (C,C)

    # 方差 -> 标准差
    var = np.diag(cov)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    # 防止除零
    denom = np.outer(std, std)
    denom = denom + eps

    pcc = cov / denom
    pcc = np.clip(pcc, -1.0, 1.0)

    if absval:
        pcc = np.abs(pcc)

    if threshold is not None:
        pcc[pcc < threshold] = 0.0

    # 对称化并数值修正
    pcc = (pcc + pcc.T) / 2.0
    return pcc


def build_initial_adj_matrix(method="distance", coords=None, names=None, data=None,
                             sigma=None, threshold=0.1, per_band=False, absval=True, eps=1e-8):
    """
    构建邻接矩阵，可选 method="distance" 或 method="pcc"。
    :param method: "distance" 或 "pcc"
    :param coords: (C,3) 仅 distance 需要
    :param names: 电极名列表（长度 C），如果提供，返回 pd.DataFrame
    :param data: EEG 数据，用于 pcc:
                 支持 (N, C, D) 或 (C, L)；
                 也支持按频带的 (N, T, C, F) 或 (N, C, T, F)（当 per_band=True 时按频带计算并合并）
    :param sigma: distance 高斯核的 sigma
    :param threshold: 阈值，小于置 0（对 distance 和 pcc 都适用）
    :param per_band: 当 data 为 4D 且为时间×频带格式时，是否对每个频带分别计算 PCC 再平均
    :param absval: 对 PCC 是否取绝对值（默认 True）
    :param eps: 数值稳定项
    :return: np.array 或 pd.DataFrame (若提供 names)
    """
#     if method == "distance":
#         if coords is None:
#             raise ValueError("method=distance 时必须提供 coords")
#         dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
#         if sigma is None:
#             sigma = np.mean(dist_matrix)
#         adj_matrix = np.exp(- dist_matrix ** 2 / (2 * sigma ** 2))

#         if threshold is not None:
#             adj_matrix[adj_matrix < threshold] = 0.0
#         adj_matrix = (adj_matrix + adj_matrix.T) / 2.0

#         if names is not None:
#             return pd.DataFrame(adj_matrix, index=names, columns=names)
#         return adj_matrix
    if method == "distance":
        if coords is None:
            raise ValueError("method=distance 时必须提供 coords")

        # 通道数
        num_nodes = coords.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    adj_matrix[i, j] = 1.0   # 自环
                else:
                    # 计算欧式距离平方
                    dis_sq = np.sum((coords[i] - coords[j]) ** 2)
                    # 权重公式 min(5/dis_sq, 1)
                    adj_matrix[i, j] = min(5.0 / dis_sq, 1.0)

        # 确保对称
        adj_matrix = (adj_matrix + adj_matrix.T) / 2.0

        # 阈值稀疏化
        if threshold is not None:
            adj_matrix[adj_matrix < threshold] = 0.0

        if names is not None:
            return pd.DataFrame(adj_matrix, index=names, columns=names)
        return adj_matrix


    elif method == "pcc":
        if data is None:
            raise ValueError("method=pcc 时必须提供 EEG data")

        arr = np.asarray(data)
        # 处理不同输入维度
        if arr.ndim == 3:
            # (N, C, D) 或 (C, N, D) 但常见为 (N, C, D)
            N, C, D = arr.shape
            mat = arr.transpose(1, 0, 2).reshape(C, -1)   # (C, N*D)
            pcc = _pcc_from_matrix(mat, eps=eps, absval=absval, threshold=threshold)

        elif arr.ndim == 2:
            # 已经是 (C, L)
            mat = arr.astype(np.float64, copy=False)
            pcc = _pcc_from_matrix(mat, eps=eps, absval=absval, threshold=threshold)

        elif arr.ndim == 4:
            # 支持两种常见组织：(N, T, C, F) 或 (N, C, T, F)
            s0, s1, s2, s3 = arr.shape
            # 如果形状为 (N, T, C, F)
            if s2 == (len(names) if names is not None else 62):
                # arr: (N, T, C, F)
                N, T, C, F = arr.shape
                if per_band:
                    mats = []
                    for f in range(F):
                        band = arr[:, :, :, f]                 # (N, T, C)
                        band_mat = band.transpose(2, 0, 1).reshape(C, -1)  # (C, N*T)
                        mats.append(_pcc_from_matrix(band_mat, eps=eps, absval=absval, threshold=None))
                    # 合并：取平均（也可改为加权和/最大值）
                    pcc = np.mean(np.stack(mats, axis=0), axis=0)
                    # 最后再施加阈值（如果指定）
                    if threshold is not None:
                        pcc[pcc < threshold] = 0.0
                        pcc = (pcc + pcc.T) / 2.0
                else:
                    # 不分频带，拼接时间维度
                    mat = arr.transpose(2, 0, 1, 3).reshape(C, -1)  # (C, N*T*F)
                    pcc = _pcc_from_matrix(mat, eps=eps, absval=absval, threshold=threshold)

            # 如果形状为 (N, C, T, F)
            elif s1 == (len(names) if names is not None else 62):
                N, C, T, F = arr.shape
                if per_band:
                    mats = []
                    for f in range(F):
                        band = arr[:, :, :, f]                  # (N, C, T)
                        band_mat = band.transpose(1, 0, 2).reshape(C, -1)  # (C, N*T)
                        mats.append(_pcc_from_matrix(band_mat, eps=eps, absval=absval, threshold=None))
                    pcc = np.mean(np.stack(mats, axis=0), axis=0)
                    if threshold is not None:
                        pcc[pcc < threshold] = 0.0
                        pcc = (pcc + pcc.T) / 2.0
                else:
                    mat = arr.transpose(1, 0, 2, 3).reshape(C, -1)  # (C, N*T*F)
                    pcc = _pcc_from_matrix(mat, eps=eps, absval=absval, threshold=threshold)
            else:
                raise ValueError(f"无法识别的 4D 结构: {arr.shape}. 期望 (N,T,C,F) 或 (N,C,T,F).")
        else:
            raise ValueError(f"不支持的数据维度 {arr.shape}")

        if names is not None:
            return pd.DataFrame(pcc, index=names, columns=names)
        return pcc

    else:
        raise ValueError("method 必须是 'distance' 或 'pcc'")
