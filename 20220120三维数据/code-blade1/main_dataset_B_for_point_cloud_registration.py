import argparse
import numpy as np
import scipy.io as sio

from utils_dataset_B_for_point_cloud_registration import loaddata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('--datasets_mat_path', help='the path of train_datasets_mat_file (default: None)',
                        default='./dataset_inference_*_2048_*.mat', metavar='None', type=str)
    parser.add_argument('--point_num', help='number of patches', default=2048, type=int, metavar='pn')
    parser.add_argument('--num_view', help='Number of viewpoint', required=False, default=3)
    parser.add_argument('--dataset', help='Dataset types', required=False, default='inference', choices=['train', 'val', 'test', 'inference'])
    arg = parser.parse_args()

    paths = arg.datasets_mat_path
    print('The datasets_mat_file path is: ', paths)

    # batch*view_num*point_num*3；batch*view_num*3*3；batch*view_num*3; batch*3;
    Viewpoints, R, T, Rotary_Center, R_init, offset_Q = loaddata(paths, arg.point_num)

    src, tgt, R_s2t, T_s2t = [], [], [], []
    for s in range(Viewpoints.shape[0]):
        View = Viewpoints[s]
        source, target, r_s2t, t_s2t = [], [], [], []
        for i in range(arg.num_view-1, 0, -1):
            target.append(View[i-1])
            source.append(View[i])
            r_s2t.append(R[s, i]@R[s, i-1].T)
            t_s2t.append((T[s, i]-Rotary_Center[s])@R[s, i]@R[s, i-1].T-(T[s, i-1]-Rotary_Center[s]))

        r_s2t.append(R[s, 0])
        t_s2t.append(T[s, 0])

        src.append(np.stack(source, axis=0))
        tgt.append(np.stack(target, axis=0))
        R_s2t.append(np.stack(r_s2t, axis=0))
        T_s2t.append(np.stack(t_s2t, axis=0))

    src = np.stack(src, axis=0) # S*(view_num-1)*point_num*3
    tgt = np.stack(tgt, axis=0) # S*(view_num-1)*point_num*3
    R_s2t = np.stack(R_s2t, axis=0) # S*view_num*3*3
    T_s2t = np.stack(T_s2t, axis=0) # S*view_num*3

    sio.savemat(f'./pcr_dataset_{arg.dataset}_{src.shape[0]}_{arg.point_num}.mat',
                {'src': src, 'tgt': tgt, 'R_s2t': R_s2t, 'T_s2t': T_s2t, 'R_init': R_init, 'offset_Q': offset_Q, 'Viewpoints': Viewpoints, 'R': R, 'T': T, 'Rotary_Center': Rotary_Center})

