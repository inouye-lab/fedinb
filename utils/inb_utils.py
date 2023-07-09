import torch
import torch.nn as nn
from autoencoders.ae_model import AE
from .metrics import part_wd, evaluate_fid_score
class wrap_enc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ae_list = []
        for dd in args.list_train_domains:
            ae = AE(args.dataset)
            ae_path = f'{args.ae_dir}/{args.dataset}/{args.ae_model}/ae-{dd}.pt'
            ae.load_state_dict(torch.load(ae_path))
            ae = ae.to(args.device)
            self.ae_list.append(ae.encoder)
            print(f'Finish loading encoder from {ae_path}')

    def forward(self, X, y):
        X = X.view(-1, 1, 28, 28)
        return self.ae_list[y](X).view(X.shape[0], -1)

class wrap_dec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ae_list = []
        self.ae_output_dim = args.ae_output_dim
        for dd in args.list_train_domains:
            ae = AE(args.dataset)
            ae_path = f'{args.ae_dir}/{args.dataset}/{args.ae_model}/ae-{dd}.pt'
            ae.load_state_dict(torch.load(ae_path))
            ae = ae.to(args.device)
            self.ae_list.append(ae.decoder)
            print(f'Finish loading decoder from {ae_path}')

    def forward(self, X, y):
        X = X.view(-1,*self.ae_output_dim)
        return self.ae_list[y](X).view(X.shape[0], -1)

def prepare_data(imgs, labels, domains, y, d, num=None):
    '''
    :param d: Wanted domain
    :param y: Wanted label
    '''

    idx = labels == y
    imgs = imgs[idx]
    labels = labels[idx]
    domains = domains[idx]

    idx = domains == d
    imgs = imgs[idx]
    labels = labels[idx]
    domains = domains[idx]

    if num:
        imgs = imgs[:num]
        labels = labels[:num]
        domains = domains[:num]

    return imgs, labels, domains

def prepare_data_domains(imgs, labels, domains, label, domain_list, train=True):
    xlist = []
    dlist = []
    for d in domain_list:
        xd, _, dd = prepare_data(imgs, labels, domains, label, d)
        xlist.append(xd)
        dlist.append(dd)
    x = torch.cat(xlist)
    x = x.view(x.shape[0], -1)
    d = torch.cat(dlist)

    if train:
        # make the number of samples to be even
        idx = int(x.shape[0] / 2) * 2
        x = x[:idx]
        d = d[:idx]

    return x, d

def inb_translate(cd, x, source_d, target_d):
    '''
    translate data
    '''
    z = cd(x,source_d)
    trans_d = torch.ones(z.shape[0]) * target_d
    x_trans = cd.inverse(z,trans_d)
    return x_trans



def eval_fid_wd_init(x,
                d,
                domain_list,
                model_enc = None,
                model_dec = None,
                fid=True,
                wd =True):
    wd_mat = torch.zeros(len(domain_list),len(domain_list))
    fid_mat = torch.zeros(len(domain_list), len(domain_list))
    for idx in domain_list:
        if model_enc is None:
            xt = x[d==idx]
        else:
            xc = x[d == idx]
            x_enc = model_enc(xc, idx)
            xt = model_dec(x_enc, idx)

        for jdx in domain_list:
            xr = x[d == jdx]
            assert torch.max(xr) <= 1 and torch.max(xt) <= 1 and torch.min(xr) >= 0\
                   and torch.min(xt) >= 0, 'Check range of output'
            if wd:
                wd_mat[idx,jdx] = part_wd(xr.cpu(),xt.cpu())
            if fid:
                fid_mat[idx, jdx] = evaluate_fid_score(
                    xr.view(-1, 1, 28, 28).cpu().detach().numpy().reshape(xr.shape[0], 28, 28, 1),
                    xt.view(-1, 1, 28, 28).cpu().detach().numpy().reshape(xt.shape[0], 28, 28, 1))

    avg_wd = torch.mean(wd_mat).item()
    avg_fid = torch.mean(fid_mat).item()

    return avg_wd, wd_mat, avg_fid, fid_mat

def eval_fid_wd(x,
                d,
                model,
                domain_list,
                model_enc = None,
                model_dec = None,
                fid=True,
                wd =True):
    wd_mat = torch.zeros(len(domain_list),len(domain_list))
    fid_mat = torch.zeros(len(domain_list), len(domain_list))
    for idx in domain_list:
        xc = x[d==idx]
        dc = d[d==idx]
        if model_enc:
            xc = model_enc(xc,idx)
        for jdx in domain_list:
            xr = x[d == jdx]
            xt = inb_translate(model,xc,dc,jdx)
            if model_dec:
                xt = model_dec(xt,jdx)
            assert torch.max(xr) <= 1 and torch.max(xt) <= 1 and torch.min(xr) >= 0\
                   and torch.min(xt) >= 0, 'Check range of output'
            if wd:
                wd_mat[idx,jdx] = part_wd(xr.cpu(),xt.cpu())
            if fid:
                fid_mat[idx, jdx] = evaluate_fid_score(
                    xr.view(-1, 1, 28, 28).cpu().detach().numpy().reshape(xr.shape[0], 28, 28, 1),
                    xt.view(-1, 1, 28, 28).cpu().detach().numpy().reshape(xt.shape[0], 28, 28, 1))

    avg_wd = torch.mean(wd_mat).item()
    avg_fid = torch.mean(fid_mat).item()

    return avg_wd, wd_mat, avg_fid, fid_mat
