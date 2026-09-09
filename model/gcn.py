"""
GCN-based multi-label classifier (ResNet-101 backbone + label-graph GCN).

Adapted from:
    Chen, Z.-M., Wei, X.-S., Wang, P., & Guo, Y. (2019).
    "Multi-Label Image Recognition with Graph Convolutional Networks." CVPR 2019.
    arXiv:1904.03582 - https://arxiv.org/abs/1904.03582
    Reference implementation: https://github.com/megvii-research/ML-GCN

Node2vec label embeddings (create_feature_matrix) follow:
    Grover, A., & Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." KDD 2016.
"""

import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import text_embedding as tx
import os
import pickle
from node2vec import Node2Vec
import networkx as nx

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()

        # generate input and output features
        # input features = i.e features per node
        self.in_features = in_features

        # output features = transformed features
        self.out_features = out_features

        # generates weight
        self.weight = Parameter(torch.Tensor(in_features, out_features)).to(device)
       
        # checks whether to generate bias or noy
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)

        # initialises weight and bias
        self.reset_parameters()

    # ensure that the params do not get too big or small (and not the same so randomness)
    def reset_parameters(self):
        # calculate std dev based on output features
        stdv = 1. / math.sqrt(self.weight.size(1))

        # giving the weights random values within a specific range
        self.weight.data.uniform_(-stdv, stdv)

        # same for bias
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # multiply input and weights (matrix multiplication)
        support = torch.matmul(input, self.weight)

        # multiple adjacency and weights
        output = torch.matmul(adj, support)

        # add bias if there is one (linear transformation)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, t, model, num_classes, allergy_data, in_channel=80):
        super(GCNResnet, self).__init__()
        
        # resnet backbone with initial layers & residual blocks
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        # output classes
        self.num_classes = num_classes

        # max pooling function -> applied to each channel
        self.pooling = nn.MaxPool2d(7, 7)

        # makes use of 2 convolutions + a LeakyRELU activation func
        self.gc1 = GraphConvolution(in_channel, 1024).to(device)
        self.gc2 = GraphConvolution(1024, 2048).to(device)
        self.relu = nn.LeakyReLU(0.2)

        # generate adjacency matrix & convert to pytorch
        _adj = gen_A(num_classes, t)
        self.A = Parameter(torch.from_numpy(_adj).float())

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):

        # passes image? through backbone and max pooling
        feature = self.features(feature)
        feature = self.pooling(feature)

        # flattens output to a 1D tensor (1 * 2048)
        feature = feature.view(feature.size(0), -1)

        # gets feature matrix and generates adj matrix
        adj = gen_adj(self.A)

        # node2vec implementation
        # inp = create_feature_matrix(adj)

        # PLM implementation
        with open('/home/ananya/Ananya_Ravi_FYP/model/gcn_pkl/feature_data.pkl', 'rb') as file:
            inp = pickle.load(file)

        # puts the input + matrix into GCN
        # static adj matrix, as the relationship between nodes doesn't change
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        # changes the dims so you can do multiplication (2048 * 10)
        x = x.transpose(0, 1)

        # multiplies to get final result (1 * 10)
        x = torch.matmul(feature, x)
        x = torch.sigmoid(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


def gcn_resnet101(t, num_classes, allergy_data, pretrained=True, in_channel=80):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(t, model, num_classes, allergy_data, in_channel=in_channel)

def gen_A(num_classes, t):

    with open('/home/ananya/Ananya_Ravi_FYP/model/gcn_pkl/cooccurrence_data.pkl', 'rb') as file:
        co_matrix, label_counts = pickle.load(file)

    _adj = co_matrix
    _nums = label_counts
    # print(_nums)
    _nums = np.array(_nums)[:, np.newaxis]

    # adj is normalised
    _adj = _adj / _nums

    # threshold is calculated for adj (binarizing matrix)
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1

    # further nornmalisation to avoid being divided by 0
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)

    # identity matrix is added to adj matrix (to account for self connection?)
    _adj = _adj + np.identity(num_classes, int)
    return _adj

def gen_adj(A):

    # degree matrix - sums up each row of adj and raises it to -0.5 (1D tensor)
    D = torch.pow(A.sum(1).float(), -0.5)
    # creates a 2D tensor by putting values along diag
    D = torch.diag(D)

    # D^-0.5 * A * D^-0.5
    # D^-0.5 prevents higher-degree nodes from dominating the GCN
    # uses twice to ensure symmetry
    # -0.5 used instead of -1 to prevent exploding/vanishing gradient
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

# older implementation
'''

NODE2VEC IMPLEMENTATION:

'''

def create_feature_matrix(adj_matrix):
    # Create a graph from the adjacency matrix
    adj_matrix_cpu = adj_matrix.cpu()
    graph = nx.from_numpy_matrix(adj_matrix_cpu.detach().numpy())

    # create node2vec embeddings
    # dim = embedding size, walk length = num nodes visited per walk, workers = multicore machines
    node2vec = Node2Vec(graph, dimensions=300, walk_length=15, num_walks=9, workers=4)

    # window = consider nodes within a certain dist, batch = process indiv or groups
    model = node2vec.fit(window=4, min_count=1, batch_words=1)
    node_embeddings = {node: model.wv[node] for node in graph.nodes()}  

    # Create feature matrix
    feature_matrix = torch.tensor(np.array([node_embeddings[node] for node in graph.nodes()])).cuda()

    return feature_matrix


