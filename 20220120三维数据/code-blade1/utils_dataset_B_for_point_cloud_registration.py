import torch
import glob
import numpy as np
import scipy.io as sio

def set_seed(SEED=2023):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED=2023)

def loaddata(paths, point_num):
    paths = glob.glob(paths)
    paths.sort()

    for i, path in enumerate(paths):
        if i == 0:
            Viewpoints = sio.loadmat(path)['viewpoints'].astype(np.float32)  # batch*view_num*point_num*3
            R = sio.loadmat(path)['R'].astype(np.float32)  # batch*view_num*3*3
            T = sio.loadmat(path)['T'].astype(np.float32)  # batch*view_num*3
            Rotary_Center = sio.loadmat(path)['Rotary_Center'].astype(np.float32)  # batch*3
            R_init = sio.loadmat(path)['R_init'].astype(np.float32)  # batch*3*3
            offset_Q = sio.loadmat(path)['offset_Q'].astype(np.float32)  # batch*3
        else:
            Viewpoints = np.concatenate([Viewpoints, sio.loadmat(path)['viewpoints'].astype(np.float32)], axis=0)
            R = np.concatenate([R, sio.loadmat(path)['R'].astype(np.float32)], axis=0)
            T = np.concatenate([T, sio.loadmat(path)['T'].astype(np.float32)], axis=0)
            Rotary_Center = np.concatenate([Rotary_Center, sio.loadmat(path)['Rotary_Center'].astype(np.float32)], axis=0)
            R_init = np.concatenate([R_init, sio.loadmat(path)['R_init'].astype(np.float32)], axis=0)
            offset_Q = np.concatenate([offset_Q, sio.loadmat(path)['offset_Q'].astype(np.float32)], axis=0)

    return Viewpoints, R, T, Rotary_Center, R_init, offset_Q