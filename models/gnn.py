import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize_embedding=False, dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant(self.bias.data, 0.0)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class ResGCN(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(ResGCN, self).__init__()
        self.res_conv1 = GraphConv(in_dim, out_dim)
        self.res_conv2 = GraphConv(out_dim, out_dim)
        self.res_conv3 = GraphConv(out_dim, out_dim)

    def forward(self, inputs, adj):
        outputs = self.res_conv1(inputs, adj)
        outputs = self.res_conv2(outputs, adj)
        outputs = self.res_conv3(outputs, adj)
        return outputs


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCN, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, inputs, adj):
        return torch.bmm(adj, self.fc(inputs))


class MultiGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=True)
        self.fc3 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, inputs, adj):
        outputs = F.relu(self.fc1(torch.bmm(adj, inputs)))
        outputs = F.relu(self.fc2(torch.bmm(adj, outputs)))
        outputs = F.relu(self.fc3(torch.bmm(adj, outputs)))
        return outputs


class GCNLayer(nn.Module):

    def __init__(self,input_features,output_features,bias=False):
        super(GCNLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):

        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self,adj,x):
        support = torch.mm(x,self.weights)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output+self.bias
        return output

class GCN(nn.Module):

    def __init__(self,input_size,hidden_size,dropout,bias=False):
        super(GCN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.gcn1 = GCNLayer(input_size,hidden_size,bias=bias)
        self.gcn2 = GCNLayer(hidden_size,hidden_size,bias=bias)
        self.gcn3 = GCNLayer(hidden_size,hidden_size,bias=bias)

        self.dropout = dropout

    def forward(self,adj,x):
        x = F.relu(self.gcn1(adj,x))
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.relu(self.gcn2(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn3(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)

        return x
