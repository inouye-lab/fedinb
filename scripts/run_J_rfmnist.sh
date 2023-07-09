#!/bin/sh
python run_translation.py --dataset 'rfmnist' --model 'histindaeinb' --nlayers 10 --K 10 --max_swd_iters 10 --hist_bins 500 --no_wandb;
python run_translation.py --dataset 'rfmnist' --model 'histindaeinb' --nlayers 10 --K 10 --max_swd_iters 30 --hist_bins 500 --no_wandb;
python run_translation.py --dataset 'rfmnist' --model 'histindaeinb' --nlayers 10 --K 10 --max_swd_iters 50 --hist_bins 500 --no_wandb;
python run_translation.py --dataset 'rfmnist' --model 'histindaeinb' --nlayers 10 --K 10 --max_swd_iters 100 --hist_bins 500 --no_wandb;
python run_translation.py --dataset 'rfmnist' --model 'histindaeinb' --nlayers 10 --K 10 --max_swd_iters 200 --hist_bins 500 --no_wandb;