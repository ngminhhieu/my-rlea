from operator import index
from scipy import spatial
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class GCN_layer(nn.Module):
    """
      Define filter layer 1/2 like in the above image
      Calculate A_hat first then,
      Input: adj_matrix with input features X
    """

    def __init__(self, first_adj_matrix, second_adj_matrix, input_shape, hidden_states):
        super(GCN_layer, self).__init__()
        self.fc = nn.Linear(input_shape, hidden_states)
        A = torch.from_numpy(first_adj_matrix).type(
            torch.LongTensor)
        I = torch.eye(A.shape[0])
        A_hat = A+I
        D = torch.sum(A_hat, axis=0)
        D = torch.diag(D)
        D_inv = torch.inverse(D)
        self.A_hat_x = torch.mm(torch.mm(D_inv, A_hat), D_inv).to(device)

        A = torch.from_numpy(second_adj_matrix).type(
            torch.LongTensor)
        I = torch.eye(A.shape[0])
        A_hat = A+I
        D = torch.sum(A_hat, axis=0)
        D = torch.diag(D)
        D_inv = torch.inverse(D)
        self.A_hat_y = torch.mm(torch.mm(D_inv, A_hat), D_inv).to(device)

    def forward(self, i, input_features):
        if i == "x":
            aggregate = torch.mm(self.A_hat_x, input_features)
        else:
            aggregate = torch.mm(self.A_hat_y, input_features)
        propagate = self.fc(aggregate)
        return propagate


class Agent(nn.Module):
    def __init__(self, first_adj_matrix, second_adj_matrix, input_shape, hidden_states, output_shape, activation='Sigmoid'):
        super(Agent, self).__init__()

        self.layer1 = GCN_layer(
            first_adj_matrix, second_adj_matrix, input_shape, hidden_states)
        self.layer2 = GCN_layer(
            first_adj_matrix, second_adj_matrix, hidden_states, output_shape)
        self.similarity_matrix = None
        self.rewards = []
        self.saved_log_probs = []

        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax()
        elif activation == 'Relu':
            self.activation = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)
        self.tanh = nn.Tanh()
        self.fc_h = nn.Linear(output_shape*2, 32)
        self.fc_p = nn.Linear(32, 2)


    def forward(self, first_embeddings, second_embeddings, states):
        x = self.layer1("x", first_embeddings)
        x = self.activation(x)
        x = self.layer2("x", x)
        G_x = self.activation(x)
        
        y = self.layer1("y", second_embeddings)
        y = self.activation(y)
        y = self.layer2("y", y)
        G_y = self.activation(y)

        lst_state_x = [G_x[s[0]] for s in states]
        lst_state_y = [G_y[s[1]] for s in states]
        g_x = torch.stack(lst_state_x)
        g_y = torch.stack(lst_state_y)
        # Linear combination
        cat_gxgy = torch.cat((g_x, g_y), 1)
        h = self.fc_h(cat_gxgy)
        h = self.dropout(h)
        h = self.sigmoid(h)
        p = self.fc_p(h)
        p = self.dropout(p)
        policy = self.softmax(p)
        return policy

class Policy(nn.Module):
    def __init__(self, input_shape):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_shape, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, first_embeddings, second_embeddings, states):
        lst_state_x = [first_embeddings[s[0]] for s in states]
        lst_state_y = [second_embeddings[s[1]] for s in states]
        g_x = torch.stack(lst_state_x)
        g_y = torch.stack(lst_state_y)
        cat_gxgy = torch.multiply(g_x, g_y)
        o = self.affine1(cat_gxgy)
        o = self.dropout(o)
        o = F.relu(o)
        action_scores = self.affine2(o)
        return F.softmax(action_scores, dim=1)