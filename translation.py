import time
from copy import deepcopy

import torch
import torchvision

from iaf import add_one_layer, IterAlignFlow

from utils.inb_utils import prepare_data_domains, inb_translate, wrap_enc, wrap_dec, eval_fid_wd, eval_fid_wd_init

import wandb

def train_indaeinb(train_loader,
                   test_loader,
                   label,
                   inb_dict,
                   tracker_dict,
                   args,
                   domain_list=[0, 1, 2, 3, 4],
                   log_interval=1,
                   verbose=True):

    nlayers = args.nlayers
    K = args.K
    max_swd_iters = args.max_swd_iters
    hist_bins = args.hist_bins

    fid = args.fid
    wd = args.wd
    vis = args.vis
    wandb = args.wandb

    # =================================================================================== #
    #                                      Prepare data                                   #
    # =================================================================================== #

    for i, (x, y, d) in enumerate(train_loader):
        train_imgs = x
        train_labels = y
        train_domains = d

    for i, (x, y, d) in enumerate(test_loader):
        test_imgs = x
        test_labels = y
        test_domains = d

    x_train, d_train = prepare_data_domains(train_imgs, train_labels, train_domains, label, domain_list)
    x_test, d_test = prepare_data_domains(test_imgs, test_labels, test_domains, label, domain_list, train=False)

    x_train = x_train.to(args.device)
    d_train = d_train.to(args.device)
    x_test = x_test.to(args.device)
    d_test = d_test.to(args.device)

    # =================================================================================== #
    #                                         Set up                                      #
    # =================================================================================== #

    start = time.time()
    model = IterAlignFlow()
    tracker = dict()
    tracker['fid'] = list()
    tracker['wd'] = list()
    tracker['nparams'] = list()

    model_enc, model_dec = wrap_enc(args), wrap_dec(args)
    z_train = torch.zeros(x_train.shape[0], args.ae_output_dim[0]*args.ae_output_dim[1]*args.ae_output_dim[2]).to(args.device)
    for d in domain_list:
        ddx = d_train == d
        z_train[ddx] = model_enc(x_train[ddx],d)
    n_params = 0

    # check before starting
    if fid or wd:
        avg_wd, wd_mat, avg_fid, fid_mat = eval_fid_wd_init(x_test,d_test,domain_list)
        print(f'Initially, the FID for digit {label} is {avg_fid}')
        print(f'Initially, the WD for digit {label} is {avg_wd}')
        if wandb:
            wandb.log({"avg_fid": avg_fid})
            wandb.log({"avg_wd": avg_wd})
        tracker['wd'].append(wd_mat)
        tracker['fid'].append(fid_mat)
        tracker['nparams'].append(n_params)

    # check after AE encoding and decoding
    if fid or wd:
        avg_wd, wd_mat, avg_fid, fid_mat = eval_fid_wd_init(x_test, d_test,domain_list,model_enc=model_enc,model_dec=model_dec)
        print(f'Initially after ae, the FID for digit {label} is {avg_fid}')
        print(f'Initially after ae, the WD for digit {label} is {avg_wd}')
        if wandb:
            wandb.log({"avg_fid": avg_fid})
            wandb.log({"avg_wd": avg_wd})
        tracker['wd'].append(wd_mat)
        tracker['fid'].append(fid_mat)
        tracker['nparams'].append(n_params)

    # =================================================================================== #
    #                                         Training                                    #
    # =================================================================================== #
    # add INB layers
    for i in range(nlayers):
        model, z_train = add_one_layer(model, z_train, d_train, K, bary_type='nb', max_swd_iters=max_swd_iters,
                                    swd_bins = hist_bins, trans_hist=args.use_hist)

        # =================================================================================== #
        #                                        Evaluation                                   #
        # =================================================================================== #
        # trans - max-K-SW
        # M*(2J(Kd+KV)+2J'KV)
        # trans - 1d bary
        # nb_params
        n_params += len(domain_list)*(
                model.layer[-1].swd_iters * 2 *
                (model.layer[-1].wT.shape[0] * model.layer[-1].wT.shape[1]
                 + hist_bins * model.layer[-1].wT.shape[1])
                + model.layer[-1].swd_extra_iters * 2 * hist_bins * model.layer[-1].wT.shape[1]) \
                   + model.layer[-1].nb_params

        if (i + 1) % log_interval == 0:
            print(f'iter {i + 1}')

            if wandb and vis:
                x_vis = x_test[d_test==0]
                d_vis = d_test[d_test==0]
                x_vis = x_vis[:10]
                d_vis = d_vis[:10]
                x_vis_list = []
                x_vis_list.append(x_vis)
                x_vis_enc = model_enc(x_vis,0)
                for vdx in domain_list:
                    x_vis_trans = inb_translate(model,x_vis_enc,d_vis,vdx)
                    x_vis_trans = model_dec(x_vis_trans,vdx)
                    x_vis_list.append(x_vis_trans)
                x_vis = torch.cat(x_vis_list)
                grid_img = torchvision.utils.make_grid(x_vis.view(-1, 1, 28, 28), nrow=10, normalize=True)
                grid_img = wandb.Image(grid_img)  # .permute(1, 2, 0))
                wandb.log({'trans_imgs': grid_img})

            if fid or wd:
                avg_wd, wd_mat, avg_fid, fid_mat = eval_fid_wd(x_test,
                                                               d_test,
                                                               model,
                                                               domain_list,
                                                               model_enc=model_enc,
                                                               model_dec=model_dec,
                                                               fid=fid,
                                                               wd=wd)
                if wandb:
                    wandb.log({"n_params": n_params})
                    wandb.log({"avg_fid": avg_fid})
                    wandb.log({"avg_wd": avg_wd})
                tracker['wd'].append(wd_mat)
                tracker['fid'].append(fid_mat)
                tracker['nparams'].append(n_params)

            if verbose:
                print(f'at iter{i + 1}, for digit {label}, {n_params} parameters has been transmitted')
                if fid or wd:
                    print(f'at iter{i + 1}, the FID for digit {label} is {avg_fid}')
                    print(f'at iter{i + 1}, the WD for digit {label} is {avg_wd}')

    print(f'fitting time: {time.time() - start} s')
    inb_dict[label] = model
    tracker_dict[label] = tracker

    return inb_dict, tracker_dict


def train_inb(train_loader,
               test_loader,
               label,
               inb_dict,
               tracker_dict,
               args,
               domain_list=[0, 1, 2, 3, 4],
               log_interval=1,
               verbose=True):

    nlayers = args.nlayers
    K = args.K
    max_swd_iters = args.max_swd_iters
    hist_bins = args.hist_bins

    fid = args.fid
    wd = args.wd
    vis = args.vis
    wandb = args.wandb

    # =================================================================================== #
    #                                      Prepare data                                   #
    # =================================================================================== #

    for i, (x, y, d) in enumerate(train_loader):
        train_imgs = x
        train_labels = y
        train_domains = d

    for i, (x, y, d) in enumerate(test_loader):
        test_imgs = x
        test_labels = y
        test_domains = d

    x_train, d_train = prepare_data_domains(train_imgs, train_labels, train_domains, label, domain_list)
    x_test, d_test = prepare_data_domains(test_imgs, test_labels, test_domains, label, domain_list, train=False)

    x_train = x_train.to(args.device)
    d_train = d_train.to(args.device)
    x_test = x_test.to(args.device)
    d_test = d_test.to(args.device)

    # =================================================================================== #
    #                                         Set up                                      #
    # =================================================================================== #

    start = time.time()
    model = IterAlignFlow()
    tracker = dict()
    tracker['fid'] = list()
    tracker['wd'] = list()
    tracker['nparams'] = list()

    z_train = model.initialize(x_train, d_train)
    n_params = 0

    # check before starting
    if fid or wd:
        avg_wd, wd_mat, avg_fid, fid_mat = eval_fid_wd_init(x_test, d_test, domain_list)
        print(f'Initially, the FID for digit {label} is {avg_fid}')
        print(f'Initially, the WD for digit {label} is {avg_wd}')
        if wandb:
            wandb.log({"avg_fid": avg_fid})
            wandb.log({"avg_wd": avg_wd})
        tracker['wd'].append(wd_mat)
        tracker['fid'].append(fid_mat)
        tracker['nparams'].append(n_params)

    # =================================================================================== #
    #                                         Training                                    #
    # =================================================================================== #
    # add INB layers
    for i in range(nlayers):
        model, z_train = add_one_layer(model, z_train, d_train, K, bary_type='nb', max_swd_iters=max_swd_iters,
                                       swd_bins=hist_bins, trans_hist=args.use_hist)

        # =================================================================================== #
        #                                        Evaluation                                   #
        # =================================================================================== #
        # trans - max-K-SW
        # M*(2J(Kd+KV)+2J'KV)
        # trans - 1d bary
        # nb_params
        n_params += len(domain_list) * (
                model.layer[-1].swd_iters * 2 *
                (model.layer[-1].wT.shape[0] * model.layer[-1].wT.shape[1]
                 + hist_bins * model.layer[-1].wT.shape[1])
                + model.layer[-1].swd_extra_iters * 2 * hist_bins * model.layer[-1].wT.shape[1]) \
                    + model.layer[-1].nb_params

        if (i + 1) % log_interval == 0:
            print(f'iter {i + 1}')
            # go back to [0,1] for evaluation, and not affecting further training
            model_temp = deepcopy(model)
            _ = model_temp.end(z_train, d_train)

            if wandb and vis:
                x_vis = x_test[d_test == 0]
                d_vis = d_test[d_test == 0]
                x_vis = x_vis[:10]
                d_vis = d_vis[:10]
                x_vis_list = []
                x_vis_list.append(x_vis)
                for vdx in domain_list:
                    x_vis_trans = inb_translate(model_temp, x_vis, d_vis, vdx)
                    x_vis_list.append(x_vis_trans)
                x_vis = torch.cat(x_vis_list)
                grid_img = torchvision.utils.make_grid(x_vis.view(-1, 1, 28, 28), nrow=10, normalize=True)
                grid_img = wandb.Image(grid_img)  # .permute(1, 2, 0))
                wandb.log({'trans_imgs': grid_img})

            if fid or wd:
                avg_wd, wd_mat, avg_fid, fid_mat = eval_fid_wd(x_test,
                                                               d_test,
                                                               model_temp,
                                                               domain_list,
                                                               fid=fid,
                                                               wd=wd)
                if wandb:
                    wandb.log({"n_params": n_params})
                    wandb.log({"avg_fid": avg_fid})
                    wandb.log({"avg_wd": avg_wd})
                tracker['wd'].append(wd_mat)
                tracker['fid'].append(fid_mat)
                tracker['nparams'].append(n_params)

            if verbose:
                print(f'at iter{i + 1}, for digit {label}, {n_params} parameters has been transmitted')
                if fid or wd:
                    print(f'at iter{i + 1}, the FID for digit {label} is {avg_fid}')
                    print(f'at iter{i + 1}, the WD for digit {label} is {avg_wd}')

    print(f'fitting time: {time.time() - start} s')
    inb_dict[label] = model
    tracker_dict[label] = tracker

    return inb_dict, tracker_dict
