# Efficient Federated Domain Translation

This repository provides the official PyTorch implementation of the following paper:

>Zhou, Z., Azam, S. S., Brinton, C., & Inouye, D. I.. Efficient federated domain translation. ICLR 2023.

It can also be used as the PyTorch implementation of the following paper (which did not support GPU originally):

>Zhou, Z., Gong, Z., Ravikumar, P., & Inouye, D. I.. Iterative alignment flows. AISTATS 2022.

## Todo
Currently, this repo only contains domain translation experiments with HistINB and HistIndAEINB. 
1. Add more investigations of federated domain translation.
2. Add code for FedINB without VW histograms.
3. Add code for domain generalization experiment.

Codes listed above will be added later. Feel free to contact the author for any specific question!

## Installation

Our models are based on PyTorch.


## Introduction 

The ```/data``` folder contains indices of MNIST and FashionMNIST to generate subset.  

The ```/iaf``` folder contains the implementation of FedINB and FedINB with VW histograms. The code is based on the [repository](https://github.com/inouye-lab/Iterative-Alignment-Flows) for paper [Iterative Alignment Flows](https://proceedings.mlr.press/v151/zhou22b/zhou22b.pdf). Here we provide the PyTorch version (the original code is implemented based on Scikit-learn and PyTorch which does not support running on GPU.) Note that in the current implementation, we don't actually create a server and clients as we prove in the paper that FedINB is equivalent to INB. We track the communication cost by computing the number of parameters that should be transmitted.

```run_translation.py``` can be used to train FedINB with RotatedMNIST and RotatedFashionMNIST. A few examples are given below.

## Implementation

### FedINB
A few examples of how to use FedINB:

Train HistINB(L10-K10-J100-V500) with RotatedMNIST.

```python run_translation.py --dataset 'rmnist' --model 'histinb' --nlayers 10 --K 10 --max_swd_iters 100 --hist_bins 500 --no_wandb```

Train HistIndAEINB(L10-K10-J100-V500) with RotatedMNIST.

```python run_translation.py --dataset 'rmnist' --model 'histindaeinb' --nlayers 10 --K 10 --max_swd_iters 100 --hist_bins 500 --no_wandb```

Train HistINB(L10-K10-J100-V500) with RotatedFashionMNIST.

```python run_translation.py --dataset 'rfmnist' --model 'histinb' --nlayers 10 --K 10 --max_swd_iters 100 --hist_bins 500 --no_wandb```

Train HistIndAEINB(L10-K10-J100-V500) with RotatedFashionMNIST.

```python run_translation.py --dataset 'rfmnist' --model 'histindaeinb' --nlayers 10 --K 10 --max_swd_iters 100 --hist_bins 500 --no_wandb```

```--nlayers``` number of INB layers

```--K``` number of dimension after projection

```--max_swd_iters``` number of maximum iterations for max-K-SW

```--hist_bins``` number of histogram bins used for VW-histogram

Currently, the computation of WD and FID does not support usage of GPU and takes a long time. If you only need the model, please add ```--no_wd --no_fid```

If you want to use Weights & Biases, then remove ```--no_wandb``` and specify the following ```--project_name <YOUR_PROJECT_NAME> --entity <YOUR_WANDB_ENTITY>```

### Federated Domain Translation
To investigate ```J``` for HistIndAEINB, execute the following scripts.

```
scripts/run_J_rmnist.sh
```
```
scripts/run_J_rfmnist.sh
```

To visualize the result, check ```notebooks/result_translation.ipynb```
## Citation
If you find this code useful, we would be grateful if you cite our [paper](https://openreview.net/forum?id=uhLAcrAZ9cJ)
```
@inproceedings{zhou2023fedinb,
  author       = {Zeyu Zhou and
                  Sheikh Shams Azam and
                  Christopher G. Brinton and
                  David I. Inouye},
  title        = {Efficient Federated Domain Translation},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  year         = {2023},
```