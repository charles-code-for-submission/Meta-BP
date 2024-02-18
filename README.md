# GraphMPL

In submission.

## Requirements

+ Python 3.10
+ PyTorch 1.1.3
+ DGL 1.0.1
+ OGB 1.3

## Dataset

The datasets used in the paper are publicly available at https://github.com/shchur/gnn-benchmark and https://ogb.stanford.edu/.

## Run

Example: to run experiments on Amazon Computers, execute the following command based on the pre-trained GNN by DGI.
+ Python amazon_sgc_3dgims_mi.py --dgipath best_dgi_computers_256_256.pkl --frozen 1 --enchid hidden_units --sparsity capacity-ratio
   
## Update

The code will be further organized and refactored upon acceptance.
