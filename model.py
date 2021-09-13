import torch as T
from torch import nn
from torch_geometric.nn import GATv2Conv, TAGConv, GCNConv
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

        # self.type_emb = nn.Linear(type_in_channels, emb_dim)
        self.type_emb = nn.Embedding(type_in_channels, emb_dim)
        self.attr_emb = nn.Linear(attr_in_channels, emb_dim)
        self.global_emb = nn.Linear(global_channels, emb_dim)

        # conv_type = GCNConv
        conv_type = TAGConv
        self.conv1 = conv_type(in_channels=emb_dim, out_channels=hidden_dim)
        self.conv2 = conv_type(in_channels=hidden_dim, out_channels=hidden_dim)
        self.conv3 = conv_type(in_channels=hidden_dim, out_channels=hidden_dim * 2)

        self.c_linear1 = nn.Linear(in_features=hidden_dim * 2 + emb_dim, out_features=hidden_dim)
        self.c_linear_emb = nn.Linear(in_features=hidden_dim, out_features=emb_dim)
        self.c_linear2 = nn.Linear(emb_dim, type_in_channels - 2)

        self.r_linear1 = nn.Linear(in_features=hidden_dim * 2 + emb_dim, out_features=hidden_dim)
        self.r_linear2 = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, type_nodes, attr_nodes, edge_index, n_type_nodes, n_attr_nodes, global_features, batch_info):
        """
        The mini-batching strategy is default.
        In order to merge the attribute nodes and type nodes to a proper graph, a index selection trick is used
        to swap the nodes' ordering.
        The rest of the network is as usual.
        :param type_nodes: The nodes for the event types
        :param attr_nodes: The nodes for the attributes of the events
        :param edge_index:
        :param n_type_nodes: Number of type nodes
        :param n_attr_nodes: Number of attribute nodes
        :param global_features: A vector of global features
        :param batch_info: Mini-batching information of Pytorch Geometric.
        :return:
        """
        x1 = self.type_emb(T.argmax(type_nodes, dim=1).long())
        x2 = self.attr_emb(attr_nodes)
        global_emb = self.global_emb(global_features)

        # We concatenate the attribute nodes to the end of the list
        # Then we sort it back to the desired position
        x = T.cat([x1, x2], dim=0)
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
        x = T.index_select(x, 0, T.tensor(gather_id).to(x.device))

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = scatter_mean(x, batch_info, dim=0)
        x = T.cat([x, global_emb], dim=1)

        c_x = self.c_linear1(x)
        c_x = F.leaky_relu(c_x)
        out_emb = self.c_linear_emb(c_x)

        out_prob = F.softmax(self.c_linear2(out_emb), dim=1)

        r_x = self.r_linear1(x)
        r_x = F.leaky_relu(r_x)
        out_time = self.r_linear2(r_x)
        return out_prob, out_emb, out_time

    def emb_y(self, y_onehot):
        return self.type_emb(F.pad(y_onehot, (0, 2)).argmax(dim=1))
