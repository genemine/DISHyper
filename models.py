import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
    
    
class resHGNNLayer(nn.Module):
    def __init__(self,nhid,dropout=0.5):
        super(resHGNNLayer, self).__init__()
        self.hgc = HGNN_conv(nhid, nhid)
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, G, *args, **kwargs):
        h = self.hgc(x, G)
        h = h + x
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
class DISHyperNet(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, num_layers=3, dropout=0.5):
        super(DISHyperNet, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_ch, n_hid)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                resHGNNLayer(n_hid, dropout=dropout)
            )
        self.outLayer = nn.Linear(n_hid, n_class)
    def forward(self, x, G, *args, **kwargs):
        x = F.relu(self.fc(x))
        x= F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = layer(x, G)
        out = self.outLayer(x) 
        return F.log_softmax(out, dim=1)
