import numpy as np
import torch
import imageio
import os
import torchvision
from torch import distributions

def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist

def get_nsamples(data_loader, N):
    x = []
    x_label = []
    n = 0    
    while n < N:
        x_next, x_next2 = next(iter(data_loader))        
        x.append(x_next)
        x_label.append(x_next2)
        #n += sum(item.size(0) for item in x_next if isinstance(item, torch.Tensor))
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    x_label = torch.cat(x_label, dim=0)[:N]
    #x = torch.cat([item for sublist in x for item in sublist if isinstance(item, torch.Tensor)], dim=0)[:N]
    return x, x_label


def count_trainable_parameters(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  return sum([np.prod(p.size()) for p in model_parameters])

def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)

def save_video(imgs, fname, as_gif=False, fps=24, quality=8):
    # convert to np.uint8
    imgs = (255 * np.clip(imgs.permute(0, 2, 3, 1).detach().cpu().numpy() / 2 + 0.5, 0, 1)).astype(np.uint8)
    imageio.mimwrite(fname, imgs, fps=fps, quality=quality)
    
    if as_gif:  # save as gif, too
        os.system(f'ffmpeg -i {fname} -r 15 '
                  f'-vf "scale=512:-1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {os.path.splitext(fname)[0] + ".gif"}')


def color_depth_map(depths, scale=None):
    """
    Color an input depth map.

    Arguments:
        depths -- HxW numpy array of depths
        [scale=None] -- scaling the values (defaults to the maximum depth)

    Returns:
        colored_depths -- HxWx3 numpy array visualizing the depths
    """

    _color_map_depths = np.array([
      [0, 0, 0],  # 0.000
      [0, 0, 255],  # 0.114
      [255, 0, 0],  # 0.299
      [255, 0, 255],  # 0.413
      [0, 255, 0],  # 0.587
      [0, 255, 255],  # 0.701
      [255, 255, 0],  # 0.886
      [255, 255, 255],  # 1.000
      [255, 255, 255],  # 1.000
    ]).astype(float)
    _color_map_bincenters = np.array([
      0.0,
      0.114,
      0.299,
      0.413,
      0.587,
      0.701,
      0.886,
      1.000,
      2.000,  # doesn't make a difference, just strictly higher than 1
    ])
  
    if scale is None:
      scale = depths.max()
  
    values = np.clip(depths.flatten() / scale, 0, 1)
    # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[
      lower_bin + 1] * alphas.reshape(-1, 1)
    return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)


# Virtual camera utils


def to_sphere(u, v):    #把2維座標轉換成3維笛卡爾座標
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])

    return s


def to_phi(u):
    return 360 * u  # 2*pi*u*180/pi

def to_theta(v):
    return np.arccos(1 - 2 * v) * 180. / np.pi

def polar_to_cartesian(r, theta, phi, deg=True): #極座標到笛卡爾座標

    if deg:
        phi = phi * np.pi / 180
        theta = theta * np.pi / 180
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    # print(f"test球面坐標點: [{cx:.4f}, {cy:.4f}, {cz:.4f}]")
    return r * np.stack([cx, cy, cz])

def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):  #從0-1範圍提取xy的值
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    # v = range_v
    #print(f"u: {u}, v: {v}")
    return to_sphere(u, v)


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):   #計算從視點(相機位置)到目標點的視角轉換 相機到世界
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)  #形狀為 (up.shape[0], 1)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))  #歸一化

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate((x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)), axis=2)

    return r_mat  #形狀(1,3,3)
