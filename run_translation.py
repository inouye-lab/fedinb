import os
import argparse
from argparse import ArgumentParser

import numpy as np

import torch
import torch.utils.data as data_utils

from data import RotationDataset
from translation import train_inb, train_indaeinb

import wandb

if __name__ == "__main__":

    parser: ArgumentParser = argparse.ArgumentParser(description='Domain Translation')

    # training
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # data
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--dataset', type=str, default='rmnist', choices=['rmnist', 'rfmnist'])
    parser.add_argument('--subset', type=str, default='med')
    parser.add_argument('--label_list',type=list,default=list(range(10)))
    parser.add_argument('--list_train_domains', type=list,
                        default=['0','15','30','45','60'],
                        help='domains used during training')

    # model
    parser.add_argument('--model', default='histindaeinb')
    parser.add_argument('--nlayers', type=int, default=10, help='L')
    parser.add_argument('--K',type=int, default=10)
    parser.add_argument('--max_swd_iters', type=int, default=200, help='J')
    parser.add_argument('--hist_bins',type=int,default=2000, help='V')

    # ae model
    parser.add_argument('--ae_dir',default='./autoencoders/saved')

    # log
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--save_dir', default='./saved/translation')
    parser.add_argument('--no_fid', action='store_true', default=False)
    parser.add_argument('--no_wd', action='store_true', default=False)
    parser.add_argument('--no_vis', action='store_true', default=False)

    # wandb
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--project_name', type=str, default='your-project-name')
    parser.add_argument('--entity',type=str, default='your-wandb-entity')




    args = parser.parse_args()

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args.device = torch.device(f"cuda:{args.cuda}" if args.cuda is not None and torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # ======================== #
    #         Logging          #
    # ======================== #
    args.fid = not args.no_fid
    args.wd = not args.no_wd
    args.vis = not args.no_vis
    args.wandb = not args.no_wandb
    args.save_dir = args.save_dir + f'/{args.dataset}/{args.model}_{args.nlayers}_{args.K}_{args.max_swd_iters}_{args.hist_bins}'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.wandb:
        wandb.init(project=args.project_name, entity=args.entity, name=args.run_name, config=args)

    # ======================== #
    #         Data             #
    # ======================== #
    train_set = RotationDataset(args.list_train_domains,
                                args.data_dir,
                                args.dataset,
                                train=True,
                                mnist_subset=args.subset)
    test_set = RotationDataset(args.list_train_domains,
                                args.data_dir,
                                args.dataset,
                                train=False,
                                mnist_subset=args.subset)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_set.data.shape[0],
                                         shuffle=True)
    test_loader = data_utils.DataLoader(test_set,
                                        batch_size=test_set.data.shape[0],
                                        shuffle=True)

    print('Finish preparing data!!!')
    print('train imgs', train_set.data.shape)
    print('test imgs', test_set.data.shape)

    # ======================== #
    #         Model            #
    # ======================== #
    if args.model.find('indae') == -1:
        args.ae_model = 'centralae'
    else:
        args.ae_model = 'indae'

    if args.model.find('hist') == -1:
        args.use_hist = False
    else:
        args.use_hist = True

    # change if using other ae_model
    if args.dataset == 'rmnist':
        args.ae_output_dim = (8,7,7)
    elif args.dataset == 'rfmnist':
        args.ae_output_dim = (32,3,3)

    # ======================== #
    #         Training         #
    # ======================== #
    inb_dict = dict()
    metric_dict = dict()
    if args.model in ['histindaeinb']:
        for label in args.label_list:
            inb_dict, tracker_dict = train_indaeinb(train_loader, test_loader,
                                                    label, inb_dict, metric_dict, args)
    elif args.model in ['histinb']:
        for label in args.label_list:
            inb_dict, tracker_dict = train_inb(train_loader, test_loader,
                                                    label, inb_dict, metric_dict, args)

    torch.save(inb_dict, f'{args.save_dir}/inb.pt')
    if args.fid and args.wd:
        torch.save(metric_dict,f'{args.save_dir}/metric.pt')
