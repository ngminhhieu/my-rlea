from operator import index
from scipy import spatial
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class Policy_GCN(nn.Module):
    def __init__(self, input_shape, hidden_states, output_shape):
        super(Policy_GCN, self).__init__()

        self.gcnconv1 = GCNConv(input_shape, hidden_states)
        self.gcnconv2 = GCNConv(hidden_states, output_shape)
        self.rewards = []
        self.saved_log_probs = []

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)
        self.fc_h = nn.Linear(output_shape, 32)
        self.fc_p = nn.Linear(32, 2)
        self.relu = nn.ReLU()


    def forward(self, graph_x, graph_y, states):
        x, edge_index_x = graph_x.x, graph_x.edge_index
        x = self.gcnconv1(x, edge_index_x)
        x = self.relu(x)
        G_x = self.gcnconv2(x, edge_index_x)
        
        y, edge_index_y = graph_y.x, graph_y.edge_index
        y = self.gcnconv1(y, edge_index_y)
        y = self.relu(y)
        G_y = self.gcnconv2(y, edge_index_y)

        lst_state_x = [G_x[s[0]] for s in states]
        lst_state_y = [G_y[s[1]] for s in states]
        g_x = torch.stack(lst_state_x)
        g_y = torch.stack(lst_state_y)
        cat_gxgy = torch.multiply(g_x, g_y)
        o = self.fc_h(cat_gxgy)
        o = self.dropout(o)
        o = self.relu(o) # Dung sigmoid bi bias như hôm trước, nên dùng relu
        p = self.fc_p(o)
        policy = self.softmax(p)
        return policy

class Policy_LR(nn.Module):
    def __init__(self, input_shape):
        super(Policy_LR, self).__init__()
        self.fc_h = nn.Linear(input_shape, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc_p = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, first_embeddings, second_embeddings, states):
        lst_state_x = [first_embeddings[s[0]] for s in states]
        lst_state_y = [second_embeddings[s[1]] for s in states]
        g_x = torch.stack(lst_state_x)
        g_y = torch.stack(lst_state_y)
        cat_gxgy = torch.multiply(g_x, g_y)
        o = self.fc_h(cat_gxgy)
        o = self.dropout(o)
        o = F.relu(o)
        action_scores = self.fc_p(o)
        return F.softmax(action_scores, dim=1)