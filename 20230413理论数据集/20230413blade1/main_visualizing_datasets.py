import argparse
import numpy as np
import open3d as o3d
import scipy.io as sio

from utils_visualizing_datasets import visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('-p', '--path', help='Path to data files', required=False, default='./Dataset_path_for_proposed_method')
    parser.add_argument('-n', '--num_view', help='Number of viewpoint', required=False, default=3)
    parser.add_argument('-d', '--dataset', help='Dataset types', required=False, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('-s', '--size', help='Number of Samples', required=False, default=2)
    parser.add_argument('-N', '--npoints', help='Number of downsampling points', required=False, default=2048)
    parser.add_argument('-H', '--height', help='Height of Cross-section', required=False,
                        default=[42.00, 53.5, 79.00], choices=[[42., 53.5, 79.], []])
    arg = parser.parse_args()

    datasets = sio.loadmat(arg.path + f'/dataset_{arg.dataset}_{arg.size}_{arg.npoints}_45.mat')
    # datasets = sio.loadmat('./Dataset_path_for_proposed_method_Noise/scale_0.175.mat')
    R = datasets['R']  # S*num_view*3*3
    T = datasets['T'] # S*num_view*3
    center = datasets['Rotary_Center']  # S*3
    viewpoints = datasets['viewpoints']  # S*num_view*npoints*3
    R_init = datasets['R_init'] # S*3*3
    offset_Q = datasets['offset_Q'] # S*3


    for s in range(arg.size):
        Viewpoints = viewpoints[s]
        '''可视化Viewpoints'''
        vis = o3d.visualization.Visualizer()  # 创建窗口对象
        vis.create_window(window_name=f'Initial Viewpoints {s + 1}')  # 设置窗口标题

        color = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]]  # 红、黄、蓝、绿
        for i in range(arg.num_view):
            visualization(Viewpoints[i], vis, point_color=color[i])

        vis.run()
        vis.destroy_window()

        '''还原检验正确性'''
        vis = o3d.visualization.Visualizer()  # 创建窗口对象
        vis.create_window(window_name=f'Alignment Viewpoints {s + 1}')  # 设置窗口标题

        for i in range(arg.num_view):
            Viewpoints[i] = (Viewpoints[i] +T[s, i, ...]- center[s, ...]) @ R[s, i, ...]
            visualization(Viewpoints[i], vis, point_color=color[i])

        vis.run()
        vis.destroy_window()

        '''还原到初始姿态'''
        vis = o3d.visualization.Visualizer()  # 创建窗口对象
        vis.create_window(window_name=f'Initial Viewpoints {s + 1}')  # 设置窗口标题

        visualization(Viewpoints.reshape(-1, 3), vis, point_color=color[0])

        '''先将模型放在质心'''
        Cross_sections = sio.loadmat('./sort_Cross_sections.mat')['Cross_sections']
        Cross_sections[:, 0] = -1 * Cross_sections[:, 0]  # 坐标朝向与测量数据保持一致

        Q = np.mean(Cross_sections, axis=0)
        Q[-1] = 0.
        Cross_sections = Cross_sections - Q

        visualization(Cross_sections@R_init[s]+offset_Q[s]+np.zeros((1, 3)), vis, point_color=color[1])

        vis.run()
        vis.destroy_window()
