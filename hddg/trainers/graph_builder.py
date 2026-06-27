# trainers/graph_builder.py
import numpy as np
from models.Electro import build_initial_adj_matrix
from models.electrode_coords import get_electrode_coords

def build_static_adj(train_data, graph_cfg, dataset_name="seed", num_electrodes=62):
    """
    graph_cfg:
        method: none | distance | pcc
        threshold
        per_band / absval (for pcc)
    """
    method = graph_cfg.get("method", "distance")
    threshold = graph_cfg.get("threshold", 0.1)

    if method == "none":
        return None

    if method == "distance":
        # 根据数据集自动获取电极坐标
        coords, names, _ = get_electrode_coords(dataset_name, num_electrodes)
        
        adj = build_initial_adj_matrix(
            method="distance",
            coords=coords,
            names=None,  # 不传递names，确保返回numpy array
            threshold=threshold
        )
        
        # 确保返回numpy array
        if hasattr(adj, 'values'):  # 如果是DataFrame
            adj = adj.values
        
        return adj

    if method == "pcc":
        per_band = bool(graph_cfg.get("per_band", True))
        absval   = bool(graph_cfg.get("absval", True))

        if isinstance(train_data, np.ndarray) and train_data.dtype == object:
            xs = [np.asarray(x) for x in list(train_data)]
            arr = np.stack(xs, axis=0)
        else:
            arr = np.asarray(train_data)

        return build_initial_adj_matrix(
            method="pcc",
            data=arr,
            names=None,
            threshold=threshold,
            per_band=per_band,
            absval=absval,
        )

    raise ValueError(f"未知的 graph.method: {method}")
