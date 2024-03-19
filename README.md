# DISHyper: Identifying new cancer genes based on the integration of annotated gene sets via hypergraph neural networks
DISHyper is a novel method to identify cancer genes based on the annotated gene sets and hypergraph neural networks (HGNN).
This repo is for the source code of "Identifying new cancer genes based on the integration of annotated gene sets via hypergraph neural networks". \
Paper Link: [DISHyper](https://www.biorxiv.org/content/10.1101/2024.01.22.576645v1)

Setup
------------------------
The setup process for DSIHyper requires the following steps:
### Download
Download DISHyper.  The following command clones the current DISHyper repository from GitHub:

    git clone https://github.com/genemine/DISHyper.git
    
### Environment Settings
> python==3.7.0 \
> scipy==1.1.0 \
> torch==1.13.0+cu117 \
> numpy==1.15.2 \
> pandas==0.23.4 \
> scikit_learn==0.19.2

GPU: GeForce RTX 2080 11G \
CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz

### Usage
(1) After this repository is downloaded and unzipped, go into the folder. 

(2) We have created examples of DISHyper for predicting pan-cancer genes, namely 'main.py'.

Assuming that you are currently in the downloaded folder, just run the following command and you will be able to built a model and make predictions:

predicting pan-cancer genes
```bash
 
python main.py ./outputFile
 
 ```
 ### Input
The input of DISHyper mainly consists of two parts, one of which is the correlation matrix of the hypergraph and the other is the labeled genes. We used two annotated gene sets in our example to predict cancer genes, but this can be easily extended to other diseases.

### Files
*main.py*: Examples of DISHyper for cancer gene identification \
*models.py*: DISHyper model \
*train_pred.py*: Training and testing functions \
*utils.py*: Supporting functions

### Cite
```
Deng, Chao, et al. "Identifying new cancer genes based on the integration of annotated gene sets via hypergraph neural networks." bioRxiv (2024): 2024-01.
```

## Contact
If you have any questions, please contact us:<br>
Chao Deng, `deng_chao@csu.edu.cn` <be>
Jianxin Wang, `jxwang@mail.csu.edu.cn` 

