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
        A_x=torch.from_numpy(first_adj_matrix).type(torch.LongTensor)
        I_x=torch.eye(A_x.shape[0])   
        A_hat_x=A_x+I_x
        D_x = torch.sum(A_hat_x,axis=0)
        D_inv_x = torch.diag(torch.pow(D_x, -0.5))  
        self.A_hat_x = torch.mm(torch.mm(D_inv_x, A_hat_x), D_inv_x).to(device)

        A_y=torch.from_numpy(second_adj_matrix).type(torch.LongTensor)
        I_y=torch.eye(A_y.shape[0])   
        A_hat_y=A_y+I_y
        D_y = torch.sum(A_hat_y,axis=0)
        D_inv_y = torch.diag(torch.pow(D_y, -0.5))  
        self.A_hat_y = torch.mm(torch.mm(D_inv_y, A_hat_y), D_inv_y).to(device)

    def forward(self, i, input_features):
        if i == "x":
            aggregate = torch.mm(self.A_hat_x, input_features)
        else:
            aggregate = torch.mm(self.A_hat_y, input_features)
        propagate = self.fc(aggregate)
        return propagate


class Agent(nn.Module):
    def __init__(self, first_adj_matrix, second_adj_matrix, input_shape, hidden_states, output_shape):
        super(Agent, self).__init__()

        self.layer1 = GCN_layer(
            first_adj_matrix, second_adj_matrix, input_shape, hidden_states)
        self.layer2 = GCN_layer(
            first_adj_matrix, second_adj_matrix, hidden_states, output_shape)
        self.rewards = []
        self.saved_log_probs = []

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)
        self.fc_h = nn.Linear(output_shape, 32)
        self.fc_p = nn.Linear(32, 2)
        self.relu = nn.ReLU()


    def forward(self, first_embeddings, second_embeddings, states):
        x = self.layer1("x", first_embeddings)
        G_x = self.relu(x)
        G_x = self.layer2("x", G_x)
        G_x = self.relu(G_x)
        
        y = self.layer1("y", second_embeddings)
        G_y = self.relu(y)
        G_y = self.layer2("y", G_y)
        G_y = self.relu(G_y)

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

class Policy(nn.Module):
    def __init__(self, input_shape):
        super(Policy, self).__init__()
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