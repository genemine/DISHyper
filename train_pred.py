import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import random
from utils import cal_auc, _generate_G_from_H_weight, getData
from models import DISHyperNet

def train_test(trainIndex, testIndex, labelFrame, incidenceMatrix, geneList, lr, epochs, dropout, n_hid, weight_decay):
    trainFrame = labelFrame.iloc[trainIndex]
    trainPositiveGene = list(trainFrame.where(trainFrame==1).dropna().index)
    positiveMatrixSum = incidenceMatrix.loc[trainPositiveGene].sum()
        
    # disease-specific hyperedge weight
    selHyperedgeIndex = np.where(positiveMatrixSum>=3)[0]
    selHyperedge = incidenceMatrix.iloc[:, selHyperedgeIndex]
    hyperedgeWeight = positiveMatrixSum[selHyperedgeIndex].values
    selHyperedgeWeightSum = incidenceMatrix.iloc[:, selHyperedgeIndex].values.sum(0)
    hyperedgeWeight = hyperedgeWeight/selHyperedgeWeightSum
        
    H = np.array(selHyperedge).astype('float')
    DV = np.sum(H * hyperedgeWeight, axis=1)
    for i in range(DV.shape[0]):
        if(DV[i] == 0):
            t = random.randint(0, H.shape[1]-1)
            H[i][t] = 0.0001
    G = _generate_G_from_H_weight(H, hyperedgeWeight)
    N = H.shape[0]
    adj = torch.Tensor(G).float()
    features = torch.eye(N).float()
    theLabels = torch.from_numpy(labelFrame.values.reshape(-1,))
        
    model = DISHyperNet(in_ch = N, n_hid = n_hid, n_class = 2, dropout = dropout)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100,200,300,400], gamma = 0.5)
        
    if torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        theLabels = theLabels.cuda()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad() 
        output = model(features, adj)
        loss_train = F.nll_loss(output[trainIndex], theLabels[trainIndex]) 
        loss_train.backward()
        optimizer.step()
        schedular.step()
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[testIndex], theLabels[testIndex])
        AUROC_val, AUPRC_val = cal_auc(output[testIndex], theLabels[testIndex])
        outputFrame = pd.DataFrame(data = output.exp().cpu().detach().numpy(), index = geneList)
    return AUROC_val, AUPRC_val, outputFrame

def trainPred(geneList, incidenceMatrix, positiveGenePath, negativeGenePath, lr, epochs, dropout, n_hid, weight_decay):
    aurocList = list()
    auprcList = list()
    evaluationRes = pd.DataFrame(index = geneList)
    for i in range(5):
        sampleIndex,label,labelFrame = getData(positiveGenePath, negativeGenePath, geneList)
        sk_X = sampleIndex.reshape([-1,1])
        sfolder = StratifiedKFold(n_splits = 5, random_state = i, shuffle = True)
        for train_index,test_index in sfolder.split(sk_X, label):
            trainIndex, testIndex, _, __ = sampleIndex[train_index], sampleIndex[test_index], label[train_index], label[test_index]
            
            AUROC_val, AUPRC_val, outputFrame = train_test(trainIndex, testIndex, labelFrame, incidenceMatrix, geneList, lr, epochs, dropout, n_hid, weight_decay)
            aurocList.append(AUROC_val.item())
            auprcList.append(AUPRC_val.item())
            evaluationRes = pd.concat([evaluationRes,outputFrame[1]], axis = 1)
    return aurocList, auprcList, evaluationRes