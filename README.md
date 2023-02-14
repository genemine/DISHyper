# DISHyper: Integration of annotated gene sets via hypergraph neural networks identifies new cancer genes
This repo is for source code of "Integration of annotated gene sets via hypergraph neural networks identifies new cancer genes". \
Paper Link: xxx
## Environment Settings
> python==3.7.0 \
> scipy==1.1.0 \
> torch==1.13.0+cu117 \
> numpy==1.15.2 \
> pandas==0.23.4 \
> scikit_learn==0.19.2

GPU: GeForce RTX 2080 11G \
CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
## Usage
(1) After this repository is downloaded and unzipped, go into the folder. 

(2) We have created examples of DISHyper for predicting pan-cancer genes, namely 'main.py'.

Assuming that you are currently in the downloaded folder, just run the following command and you will be able to built a model and make predictions:

predicting pan-cancer genes
```bash
 
python main.py ./outputFile
 
 ```
 
## Files
main.py: Examples of DISHyper for cancer gene identification
models.py: DISHyper model
train_pred.py: Training and testing functions
utils.py: Supporting functions

## Cite
```

```
## Contact
If you have any questions, please contact us:<br>
Chao Deng, deng_chao@csu.edu.cn <br>
 
