from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

def fc_matrix_to_graph(fc_matrix: np.ndarray, threshold: float = 0.0) -> Data:
    """Convert functional connectivity matrix to PyTorch Geometric Data object."""
    n_rois = fc_matrix.shape[0]

    edge_index_list = []
    edge_weight_list = []

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            if abs(fc_matrix[i, j]) > threshold:
                edge_index_list.append([i, j])
                edge_index_list.append([j, i])
                edge_weight_list.append(fc_matrix[i, j])
                edge_weight_list.append(fc_matrix[i, j])

    if not edge_index_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.tensor([], dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32)

    x = torch.ones((n_rois, 1), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    return data


print("Loading preprocessed data...")
data = np.load('processed/rest_meta_mdd_fc.npz', allow_pickle=True)
fc_matrices = data['fc_matrices']
labels = data['labels']
print(f"Loaded {len(fc_matrices)} samples, {len(labels)} labels")

print("\nConverting to graphs (sample 10 only for speed)...")
for i in range(min(10, len(fc_matrices))):
    print(f"  Sample {i+1}/10...", end=" ", flush=True)
    graph = fc_matrix_to_graph(fc_matrices[i])
    print(f"nodes={graph.x.shape[0]}, edges={graph.edge_index.shape[1]}")

print("\nGraph conversion test passed!")
