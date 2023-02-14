import pandas as pd
import sys, os, random
import numpy as np
import scipy.sparse as sp
from train_pred import trainPred
from utils import processingIncidenceMatrix
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
 

if __name__ == "__main__":    
    _, outputPath = sys.argv
    lr = 5e-3
    dropout = 0.2
    weight_decay = 5e-6
    epochs = 500
    n_hid = 256
    
    positiveGenePath = r'./Data/796true.txt'
    negativeGenePath = r'./Data/2187false.txt'
    geneList = pd.read_csv(r'./Data/geneList.csv', header=None)
    geneList = list(geneList[0].values)
    incidenceMatrix = processingIncidenceMatrix(geneList)
    
    aurocList, auprcList, evaluationRes = trainPred(geneList, incidenceMatrix, positiveGenePath,
                                          negativeGenePath, lr, epochs, dropout, n_hid, weight_decay) 
    predRes = evaluationRes.sum(1).sort_values(ascending = False) / 25
    predRes.to_csv(outputPath,sep='\t', header = False)
    print(np.mean(aurocList)) # 0.936
    print(np.mean(auprcList)) # 0.892