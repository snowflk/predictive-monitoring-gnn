import torch as T
from torch import nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_scatter import scatter_mean


class PMGCN(nn.Module):
    def __init__(self, type_in_channels=6,
                 attr_in_channels=8,
                 global_channels=2,
                 emb_dim=16,
                 hidden_dim=32):
        super().__init__()
        self.type_in_channels = type_in_channels
        self.attr_in_channels = attr_in_channels
        self.global_channels = global_channels
        self.emb_dim = emb_dim

        self.type_emb = nn.Linear(type_in_channels, emb_dim)
        self.attr_emb = nn.Linear(attr_in_channels, emb_dim)
        self.global_emb = nn.Linear(global_channels, emb_dim)

        self.conv1 = GATv2Conv(in_channels=emb_dim, out_channels=hidden_dim)
        self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.conv3 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim + emb_dim, out_features=emb_dim)
        self.sm = nn.Linear(emb_dim, type_in_channels)

    def forward(self, type_nodes, attr_nodes, edge_index, n_type_nodes, n_attr_nodes, global_features, batch_info):
        x1 = self.type_emb(type_nodes)
        x2 = self.attr_emb(attr_nodes)
        global_emb = self.global_emb(global_features)
        x = T.cat([x1, x2], dim=0)  # Cat along node dimension
        gather_id = []
        curr_type_idx = 0
        curr_attr_idx = x1.shape[0]
        batch_size = len(n_attr_nodes)
        for i in range(batch_size):
            for j in range(n_type_nodes[i]):
                gather_id.append(curr_type_idx + j)
            for j in range(n_attr_nodes[i]):
                gather_id.append(curr_attr_idx + j)
            curr_type_idx += n_type_nodes[i]
            curr_attr_idx += n_attr_nodes[i]
        x = T.index_select(x, 0, T.tensor(gather_id))

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = scatter_mean(x, batch_info, dim=0)

        out_emb = self.linear(T.cat([x, global_emb], dim=1))
        out_prob = F.softmax(self.sm(out_emb), dim=1)
        return out_prob, out_emb

    def emb_y(self, y_onehot):
        return self.type_emb(y_onehot)


model = PMGCN(1, 2, 3)
