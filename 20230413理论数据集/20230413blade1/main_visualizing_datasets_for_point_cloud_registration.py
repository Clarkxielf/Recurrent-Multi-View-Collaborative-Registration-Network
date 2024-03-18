import argparse
import numpy as np
import open3d as o3d
import scipy.io as sio

from utils_visualizing_datasets_for_point_cloud_registration import visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('--pcr_dataset_path', help='Dataset path for point cloud registration', required=False,
                        default='./Dataset_path_for_point_cloud_registration')
    parser.add_argument('--dataset', help='Dataset types', required=False, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--npoints', help='Number of downsampling points', required=False, default=2048)
    parser.add_argument('--sample_num', help='Number of sample', required=False, default=2)
    parser.add_argument('-n', '--view_num', help='Number of viewpoint', required=False, default=3)
    arg = parser.parse_args()

    datasets = sio.loadmat(arg.pcr_dataset_path + f'/pcr_dataset_{arg.dataset}_{arg.sample_num}_{arg.npoints}.mat')
    # datasets = sio.loadmat('./Dataset_path_for_point_cloud_registration_Noise/pcr_scale_0.175.mat')
    src = datasets['src']  # S*(view_num-1)*point_num*3
    tgt = datasets['tgt']  # S*(view_num-1)*point_num*3
    R_s2t = datasets['R_s2t']  # S*view_num*3*3
    T_s2t = datasets['T_s2t']  # S*view_num*3
    R_init = datasets['R_init'] # S*3*3
    offset_Q = datasets['offset_Q'] # S*3
    Viewpoints = datasets['Viewpoints'] # batch*view_num*point_num*3
    R = datasets['R'] # batch*view_num*3*3
    T = datasets['T'] # batch*view_num*3
    Rotary_Center = datasets['Rotary_Center'] # batch*3

    color = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]]  # 红、黄、蓝、绿
    for s in range(2):
        source = src[s]
        target = tgt[s]
        r_s2t = R_s2t[s]
        t_s2t = T_s2t[s]
        for i in range(arg.view_num-1):
            if i==0:
                blade_profile = source[i]@r_s2t[i]+t_s2t[i][None, ...]
                blade_profile = np.concatenate([blade_profile, target[i]], axis=0)

                vis = o3d.visualization.Visualizer()  # 创建窗口对象
                vis.create_window(window_name=f'{arg.view_num-i}~{arg.view_num-i-1}')  # 设置窗口标题
                visualization(blade_profile, vis, point_color=color[i])
                vis.run()
                vis.destroy_window()
            else:
                blade_profile = blade_profile @ r_s2t[i]+t_s2t[i][None, ...]
                blade_profile = np.concatenate([blade_profile, target[i]], axis=0)

                vis = o3d.visualization.Visualizer()  # 创建窗口对象
                vis.create_window(window_name=f'{arg.view_num-i}~{arg.view_num-i - 1}')  # 设置窗口标题
                visualization(blade_profile, vis, point_color=color[i])
                vis.run()
                vis.destroy_window()


        vis = o3d.visualization.Visualizer()  # 创建窗口对象
        vis.create_window(window_name=f'blade_profile & Cross_sections')  # 设置窗口标题

        '''还原视场'''
        View = Viewpoints[s]
        center = Rotary_Center[s]
        R_ = R[s]
        T_ = T[s]
        for k in range(arg.view_num):
            View[k] = (View[k] + T_[k][None, ...]- center[None, ...]) @ R_[k]
        View_target = View.reshape(-1, 3)
        visualization(View_target, vis, point_color=color[1])

        blade_profile = (blade_profile + T_[0][None, ...] - center[None, ...]) @ R_[0]
        visualization(blade_profile+np.zeros((1, 3)), vis, point_color=color[0])

        vis.run()
        vis.destroy_window()

        print(f'{np.mean(blade_profile-View_target, axis=0)}\n')
