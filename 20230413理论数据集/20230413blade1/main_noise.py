import os
import argparse
import glob
import numpy as np
import scipy.io as sio

from utils_noise import RandomJitter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')

    parser.add_argument('--root_path', help='the path of mat_data (default: None)',
                        default='./Dataset_path_for_proposed_method', metavar='None', type=str)
    parser.add_argument('--test_datasets_mat_path', help='the path of test_datasets_mat_file (default: None)',
                        default='dataset_test_*_2048_*.mat', metavar='None', type=str)
    parser.add_argument('--test_out_dir', help='the file of the testing results',
                        default='./Dataset_path_for_proposed_method_Noise', type=str)
    parser.add_argument('--scale', help='Standard deviation (spread or "width") of the distribution. Must be non-negative',
                        default=0.175, type=float)
    arg = parser.parse_args()

    path = arg.root_path + '/' + arg.test_datasets_mat_path
    paths = glob.glob(path)
    paths.sort()
    Viewpoints, R, T, Rotary_Center, R_init, offset_Q = [], [], [], [], [], []
    for i, path in enumerate(paths):
        Viewpoints.append(sio.loadmat(path)['viewpoints'].astype(np.float32))  # batch*view_num*point_num*3
        R.append(sio.loadmat(path)['R'].astype(np.float32))  # batch*view_num*3*3
        T.append(sio.loadmat(path)['T'].astype(np.float32))  # batch*view_num*3
        Rotary_Center.append(sio.loadmat(path)['Rotary_Center'].astype(np.float32))  # batch*3
        R_init.append(sio.loadmat(path)['R_init'].astype(np.float32))  # batch*3*3
        offset_Q.append(sio.loadmat(path)['offset_Q'].astype(np.float32))  # batch*3

    Viewpoints = np.concatenate(Viewpoints, axis=0)
    R = np.concatenate(R, axis=0)
    T = np.concatenate(T, axis=0)
    Rotary_Center = np.concatenate(Rotary_Center, axis=0)
    R_init = np.concatenate(R_init, axis=0)
    offset_Q = np.concatenate(offset_Q, axis=0)

    Viewpoints = RandomJitter(Viewpoints, arg.scale)

    result_path = arg.test_out_dir
    # if os.path.exists(result_path):
    #     os.system('rm -rf ' + result_path)
    # os.makedirs(result_path)

    sio.savemat(result_path+f'/scale_{arg.scale}.mat',
                {'viewpoints': Viewpoints, 'R': R, 'T':T, 'Rotary_Center': Rotary_Center, 'R_init': R_init, 'offset_Q': offset_Q})