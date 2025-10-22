# AusRec

This is the official implementation of the following paper:

Automatic Self-supervised Learning for Social Recommendations

<div align="center">
  <img src="https://github.com/hexin5515/AusRec/blob/main/Image/AusRec.jpg" width="1600px"/>
</div>

## Environment Setup

**Required Dependencies** :

* torch>=1.9.1
* python>=3.8.1
* scikit-learn>=1.4.1
* networkx>=2.7
* tqdm
* wandb

## Quick Start

**Lastfm Dataset**

The main experiments:
```
Training and testing
python main.py --bpr_batch 2048 --recdim 192 --dataset lastfm --model lgn --layer 3 --epochs 1000
```
