import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from functools import partial

import matplotlib.pyplot as plt

from .run_nerf_helpers_mod import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

relu = partial(F.relu, inplace=True)            # saves a lot of memory


def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs, label):
        label_oftype = label[:,0]
        return torch.cat([fn(inputs[i:i+chunk], label_oftype) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, label, embed_fn, embeddirs_fn, features=None, netchunk=1024*64):   #輸出rgb and sigma
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) #524288 3
    embedded = embed_fn(inputs_flat)
    #print(f"0embedded.shape: {embedded.shape}") 524288 63
    if features is not None:
        # expand features to shape of flattened inputs  524288 256
        features = features.unsqueeze(1).expand(-1, inputs.shape[1], -1).flatten(0, 1)
        features_shape = features
        # feat_dim_appearance =256
        # features_shape = features[:, :-feat_dim_appearance]
        # features_appearance = features[:, -feat_dim_appearance:]

        embedded = torch.cat([embedded, features_shape], -1)
        # print(f"features: {features_shape}")  319

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape) #8192 64 3
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  #524288 3
        embedded_dirs = embeddirs_fn(input_dirs_flat) #524288 27
        embedded = torch.cat([embedded, embedded_dirs], -1)  #524288 346
        # if features_appearance is not None:
        #     features_appearance = features_appearance[:embedded.shape[0], :]
        #     embedded = torch.cat([embedded, features_appearance], dim=-1)

    outputs_flat = batchify(fn, netchunk)(embedded, label)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])  #8192 64 4
    return outputs


def batchify_rays(rays_flat, label, chunk=1024*32, **kwargs):  #批次render rays

    all_ret = {}
    features = kwargs.get('features')
    for i in range(0, rays_flat.shape[0], chunk):
        if features is not None:
            kwargs['features'] = features[i:i+chunk]
        ret= render_rays(rays_flat[i:i+chunk],label,  **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, label, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays #8192 3
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # if c2w_staticcam is not None:
        #     # special case to visualize effect of viewdirs
        #     rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float() #8192 3

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    # save_rays_torch(rays_o, rays_d)

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    #print("rays values:", rays) #torch.Size([8192, 8])
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) #torch.Size([8192, 11])
        #print("rays values:", rays)

    # Expand features to shape of rays
    if kwargs.get('features') is not None:
        bs = kwargs['features'].shape[0]
        N_rays = sh[0] // bs
        kwargs['features'] = kwargs['features'].unsqueeze(1).expand(-1, N_rays, -1).flatten(0, 1)

    # Render and reshape
    all_ret = batchify_rays(rays, label, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch += args.feat_dim
    # input_ch += args.feat_dim - 256
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    # input_ch_views += args.feat_dim_appearance
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, numclasses=args.num_class)
    grad_vars = list(model.parameters())
    named_params = list(model.named_parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += list(model_fine.parameters())
        named_params = list(model_fine.named_parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, label, features: run_network(inputs, viewdirs, network_fn, label,
                                                                                  features=features,
                                                                                  embed_fn=embed_fn,
                                                                                  embeddirs_fn=embeddirs_fn,
                                                                                  netchunk=args.netchunk,
                                                                                #   feat_dim_appearance=args.
                                                                                #   feat_dim_appearance
                                                                                  )

    render_kwargs_train = {             
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'ndc': False,
        'lindisp': False,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, grad_vars, named_params


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
    """ A helper function for `render_rays`.
    """
    raw2alpha = lambda raw, dists, act_fn=relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1] #採樣點之間的距離
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)+1e-10))     # add eps to avoid division by zero
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                label,
                network_fn,
                network_query_fn,
                N_samples,
                features=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn, label, features)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

#     if N_importance > 0:

#         rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

#         z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
#         z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
#         z_samples = z_samples.detach()

#         z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
#         pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

#         run_fn = network_fn if network_fine is None else network_fine
# #         raw = run_network(pts, fn=run_fn)
#         raw = network_query_fn(pts, viewdirs, run_fn, label, features)

#         rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    # if retraw:
    #     ret['raw'] = raw

    # if N_importance > 0:
    #     ret['rgb0'] = rgb_map_0
    #     ret['disp0'] = disp_map_0
    #     ret['acc0'] = acc_map_0
    #     ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def save_rays_torch(rays_o, rays_d, save_path="rays_data.pt"):
    # 創建一個字典來儲存兩個張量
    rays_data = {
        "rays_o": rays_o,
        "rays_d": rays_d
    }
    # 保存到檔案
    torch.save(rays_data, save_path)
    print(f"射線數據已儲存至 {save_path}")