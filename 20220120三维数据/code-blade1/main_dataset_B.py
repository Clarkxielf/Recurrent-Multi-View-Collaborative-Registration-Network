import argparse
import pandas as pd
import numpy as np
import scipy.io as sio

from utils_dataset_B import data3D, truncation, farthest_subsample_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('-p', '--path', help='Path to data files', required=False, default='.')
    parser.add_argument('-d', '--dataset', help='Dataset types', required=False, default='inference', choices=['train', 'val', 'test', 'inference'])
    parser.add_argument('-s', '--size', help='Number of Samples', required=False, default=2)
    parser.add_argument('-n', '--npoints', help='Number of downsampling points', required=False, default=2048)
    parser.add_argument('-c', '--center', help='Rotary_Center', required=False, default=[3.0, -5.0, 0.])
    parser.add_argument('-a', '--angle', help='Motion Parameters (Angle)', required=False, default=[0, 60, 160])
    parser.add_argument('-t', '--translation', help='Motion Parameters', required=False, default=[[0, 0, 0], [0, 0, 0], [5, -5, 0]])
    parser.add_argument('-H', '--height', help='Height of cross-sections (mm)', required=False, default=[42., 53.5, 79.])
    parser.add_argument('-N', '--num_view', help='Number of viewpoint', required=False, default=3)
    arg = parser.parse_args()

    '''提取3D数据'''
    V1 = pd.read_csv(arg.path + '/S1.csv', header=None, low_memory=False).values
    V2 = pd.read_csv(arg.path + '/S2.csv', header=None, low_memory=False).values
    V3 = pd.read_csv(arg.path + '/S3.csv', header=None, low_memory=False).values

    V1 = data3D(V1)
    V2 = data3D(V2)
    V3 = data3D(V3)

    '''调整Z向坐标'''
    shift = np.array([[0, 0, 5.]]) #以Z=5为扫描初始位置 [37., 48.5, 74.]
    V1 = V1 + shift
    V2 = V2 + shift
    V3 = V3 + shift

    '''截取特定截面'''
    width = 1.
    V1 = truncation(V1, H=arg.height, width=width)
    V2 = truncation(V2, H=arg.height, width=width)
    V3 = truncation(V3, H=arg.height, width=width)

    '''下采样到固定点数'''
    viewpoints = []
    for s in range(arg.size):
        print(f'Samples are being made: {s + 1}/{arg.size}')
        sample_v1 = farthest_subsample_points(V1, arg.npoints)
        sample_v2 = farthest_subsample_points(V2, arg.npoints)
        sample_v3 = farthest_subsample_points(V3, arg.npoints)

        viewpoints.append(np.stack([sample_v1, sample_v2, sample_v3], axis=0))

    viewpoints = np.stack(viewpoints, axis=0) # size*num_view*npoints*3

    '''R, t'''
    R = []
    t = []
    Rotary_Center = []
    theta = [np.pi * angle / 180 for angle in arg.angle]
    for s in range(arg.size):
        R_v1 = np.array([[np.cos(theta[0]), -np.sin(theta[0]), 0], [np.sin(theta[0]), np.cos(theta[0]), 0], [0, 0, 1]])
        R_v2 = np.array([[np.cos(theta[1]), -np.sin(theta[1]), 0], [np.sin(theta[1]), np.cos(theta[1]), 0], [0, 0, 1]])
        R_v3 = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0], [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])
        R.append(np.stack([R_v1, R_v2, R_v3], axis=0))

        t_v1 = np.array(arg.translation[0])
        t_v2 = np.array(arg.translation[1])
        t_v3 = np.array(arg.translation[2])
        t.append(np.stack([t_v1, t_v2, t_v3], axis=0))

        Rotary_Center.append(np.array(arg.center))

    R = np.stack(R, axis=0) # size*num_view*3*3
    t = np.stack(t, axis=0) # size*num_view*3
    Rotary_Center = np.stack(Rotary_Center, axis=0) # size*3

    T = np.zeros((arg.size, arg.num_view, 3))
    R_init = np.stack([np.eye(3) for i in range(arg.size)], axis=0)
    offset_Q = np.zeros((arg.size, 3))

    viewpoints = viewpoints + t[:, :, np.newaxis, :] # v'
    sio.savemat(arg.path + f'/dataset_{arg.dataset}_{arg.size}_{arg.npoints}_0.mat',
                {'viewpoints': viewpoints, 'R': R, 'Rotary_Center': Rotary_Center, 'T': T, 'R_init': R_init, 'offset_Q': offset_Q})
