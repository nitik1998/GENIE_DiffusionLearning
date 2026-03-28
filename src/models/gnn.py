import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, SAGEConv, global_mean_pool, global_max_pool

def build_knn_graph(points: torch.Tensor, label: int, k: int = 8) -> Data:
    """
    Construct a k-NN graph from a point cloud using Cartesian distances in (eta, phi).

    Node features use the active repo convention:
    [eta_norm, phi_norm, E_Tracks, E_ECAL, E_HCAL, r_centroid]
    Edge features are deterministic spatial relations:
    [delta_eta, delta_phi, distance, delta_intensity].
    """
    pos = points[:, :2]
    n_nodes = pos.size(0)

    if n_nodes < 2:
        # Create a self-loop dummy graph if less than 2 nodes
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr = torch.zeros((1, 4), dtype=torch.float32)
        return Data(x=points, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))

    dist = torch.cdist(pos, pos)
    k_actual = min(k + 1, n_nodes)
    _, indices = dist.topk(k_actual, largest=False)

    row = torch.arange(n_nodes).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    col = indices.reshape(-1)

    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    src = pos[edge_index[0]]
    dst = pos[edge_index[1]]
    delta = dst - src
    distance = torch.norm(delta, dim=1, keepdim=True)
    node_intensity = points[:, 2:5].sum(dim=1, keepdim=True)
    intensity_delta = node_intensity[edge_index[1]] - node_intensity[edge_index[0]]
    edge_attr = torch.cat([delta, distance, intensity_delta], dim=1)

    return Data(
        x=points,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
    )


class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE-based GNN for Jet Classification.
    Aligns closely with standard message-passing networks evaluated in past GSoC projects.
    """

    def __init__(self, in_channels: int = 5, hidden: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        
        # Message Passing Layers
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden * 2)
        self.conv3 = SAGEConv(hidden * 2, hidden * 4)
        
        self.pool_dim = hidden * 4 * 2 # Concat of Mean + Max pool
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.pool_dim, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden, 1) # Binary classification output (logits)
        )

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Node embeddings via message passing
        h = self.conv1(x, edge_index)
        h = torch.nn.functional.leaky_relu(h, 0.2)
        
        h = self.conv2(h, edge_index)
        h = torch.nn.functional.leaky_relu(h, 0.2)
        
        h = self.conv3(h, edge_index)
        h = torch.nn.functional.leaky_relu(h, 0.2)

        # Graph-level Readout (Global Pooling)
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_graph = self.encode_graph(x, edge_index, batch)

        # Final classification
        logits = self.classifier(h_graph)
        return logits.squeeze(-1)


class EdgeConvClassifier(nn.Module):
    """
    EdgeConv-based GNN for sparse detector graphs.
    This follows the common-task intuition that local geometric relations between
    nearby active detector hits are especially informative.
    """

    def __init__(self, in_channels: int = 6, hidden: int = 64, dropout: float = 0.4) -> None:
        super().__init__()

        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_channels, hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            aggr="max",
        )
        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            aggr="max",
        )

        self.pool_dim = hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.pool_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_graph = self.encode_graph(x, edge_index, batch)
        logits = self.classifier(h_graph)
        return logits.squeeze(-1)
