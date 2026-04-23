from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GATClassifier(nn.Module):
    """Graph Attention Network for binary classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATv2Conv(
            in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True
        )
        self.gat2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.fc = nn.Linear(hidden_channels * num_heads, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.fc(x)
        return x


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


def train_epoch(
    model: GATClassifier,
    train_data: list[tuple[Data, int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for graph, label in train_data:
        graph = graph.to(DEVICE)
        label_tensor = torch.tensor([label], dtype=torch.long).to(DEVICE)
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out, label_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(train_data), 1)


@torch.no_grad()
def evaluate(
    model: GATClassifier,
    val_data: list[tuple[Data, int]],
    criterion: nn.Module,
) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for graph, label in val_data:
        graph = graph.to(DEVICE)
        label_tensor = torch.tensor([label], dtype=torch.long).to(DEVICE)
        out = model(graph.x, graph.edge_index)
        loss = criterion(out, label_tensor)
        total_loss += loss.item()

        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)

        all_preds.append(preds.cpu().item())
        all_labels.append(label)
        all_probs.append(probs[0, 1].cpu().item())

    avg_loss = total_loss / max(len(val_data), 1)
    acc = accuracy_score(all_labels, all_preds)
    metrics = {
        'accuracy': acc,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
    }
    return avg_loss, acc, metrics


def main() -> None:
    set_seed(SEED)

    print('Loading data...')
    data = np.load('processed/rest_meta_mdd_fc.npz', allow_pickle=True)
    fc_matrices = data['fc_matrices'][:100]  # Only first 100 samples
    labels = data['labels'][:100]
    print(f'Loaded {len(fc_matrices)} graphs (sample subset for testing)')

    print('Converting to graphs...')
    graphs = [fc_matrix_to_graph(fc) for fc in fc_matrices]
    dataset = [(g.to(DEVICE), int(y)) for g, y in zip(graphs, labels)]

    model = GATClassifier(in_channels=1, hidden_channels=64, num_heads=4, dropout=0.2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    print(f'Model on device: {DEVICE}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Train samples: {len(labels)}\n')

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(graphs, labels)):
        print(f'Fold {fold + 1}/3')

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        for epoch in range(10):
            train_loss = train_epoch(model, train_data, optimizer, criterion)
            val_loss, val_acc, metrics = evaluate(model, val_data, criterion)
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch {epoch + 1:2d} | train_loss {train_loss:.4f} | '
                      f'val_loss {val_loss:.4f} | acc {val_acc:.4f} | f1 {metrics["f1"]:.4f}')

        _, _, final_metrics = evaluate(model, val_data, criterion)
        fold_results.append(final_metrics)
        print(f'  Final: Acc {final_metrics["accuracy"]:.4f}, F1 {final_metrics["f1"]:.4f}\n')

    print('=' * 60)
    print('Cross-validation results (3-fold on 100 samples):')
    for fold, result in enumerate(fold_results):
        print(f'Fold {fold + 1}: Acc {result["accuracy"]:.4f} | F1 {result["f1"]:.4f}')

    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])
    print(f'Mean: Acc {mean_acc:.4f}, F1 {mean_f1:.4f}')


if __name__ == '__main__':
    main()
