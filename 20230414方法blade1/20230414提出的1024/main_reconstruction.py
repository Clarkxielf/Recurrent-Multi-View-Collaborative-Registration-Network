import os
import argparse
import pandas as pd
import numpy as np
import open3d as o3d

from utils_train_val.common import visualization, visualization2, truncation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('-p', '--subpath', help='Path to data files', required=False, default='20220120三维数据/code-blade1')
    parser.add_argument('-H', '--height', help='Height of cross-sections (mm)', required=False, default=[42., 53.5, 79.])
    parser.add_argument('-a', '--angle', help='Motion Parameters (Angle)', required=False, default=[0, 60, 160])
    parser.add_argument('-t', '--translation', help='Motion Parameters', required=False, default=[[0, 0, 0], [0, 0, 0], [5, -5, 0]])
    parser.add_argument('--experiment', help='Experiment name', required=False, default='3D_reconstruction_our_method')
    parser.add_argument('--view_num', help='Number of viewpoint', required=False, default=3)
    parser.add_argument('-c', '--center', help='Rotary_Center', required=False, default=[3.8978, -5.7470,  0.0000])
    arg = parser.parse_args()

    result_path = './results_' + arg.experiment
    if os.path.exists(result_path):
        os.system('rm -rf ' + result_path)
    os.makedirs(result_path)

    path = os.getcwd().split('\\20230414方法blade1')[0] + '/' + arg.subpath
    '''加载原始数据并可视化'''
    vis = o3d.visualization.Visualizer()  # 创建窗口对象
    vis.create_window(window_name='Initial Viewpoints')  # 设置窗口标题

    V1 = pd.read_csv(path + '/S1.csv', header=None, low_memory=False).values
    V1 = visualization2(V1, vis, point_color=[1, 0, 0])  # 红色

    V2 = pd.read_csv(path + '/S2.csv', header=None, low_memory=False).values
    V2 = visualization2(V2, vis, point_color=[1, 1, 0])  # 黄色

    V3 = pd.read_csv(path + '/S3.csv', header=None, low_memory=False).values
    V3 = visualization2(V3, vis, point_color=[0, 0, 1])  # 蓝色

    vis.run()
    vis.destroy_window()

    '''调整Z向坐标'''
    shift = np.array([[0, 0, 5.]]) #以Z=5为扫描初始位置 [37., 48.5, 74.]
    V1 = V1 + shift
    V2 = V2 + shift
    V3 = V3 + shift

    '''截取未拼接型面数据'''
    width = 1.
    Partial_V1 = truncation(V1, arg.height[2]+width, arg.height[0]-12*width)
    Partial_V2 = truncation(V2, arg.height[2]+width, arg.height[0]-12*width)
    Partial_V3 = truncation(V3, arg.height[2]+width, arg.height[0]-12*width)


    vis = o3d.visualization.Visualizer()  # 创建窗口对象
    vis.create_window(window_name='Partial_Vs')  # 设置窗口标题

    visualization(Partial_V1, vis, point_color=[1, 0, 0])  # 红色
    visualization(Partial_V2, vis, point_color=[1, 1, 0])  # 黄色
    visualization(Partial_V3, vis, point_color=[0, 0, 1])  # 蓝色

    vis.run()
    vis.destroy_window()


    Partial_Vs = [Partial_V1, Partial_V2, Partial_V3]
    # for i in range(arg.view_num):
    #     Partial_V = Partial_Vs[i]
    #     test_cache_file = result_path + f'/result_Partial_V{i+1}.txt'
    #     cf = open(test_cache_file, 'a+')
    #     for j in Partial_V:
    #         cf.write(str(j)[1:-1] + ' ')
    #         cf.write('\n')
    #     cf.close()



    '''截取未拼接型线数据'''
    Partial_Vs = np.concatenate([Partial_V1, Partial_V2, Partial_V3], axis=0)
    cross_sections = []
    for h in arg.height:
        Z_max = h + 0.13
        Z_min = h - 0.14
        index0 = Partial_Vs[:, 2] > Z_min
        index1 = Partial_Vs[:, 2] < Z_max

        S_h = Partial_Vs[index0 & index1]
        '''下采样'''
        num = S_h.shape[0]//1
        d = S_h.shape[0] // num
        d_index = [d * i for i in range(num)]
        cross_sections.append(S_h[d_index])

    cross_sections = np.concatenate(cross_sections, axis=0)

    # test_cache_file = result_path + f'/result_cross_sections.txt'
    # cf = open(test_cache_file, 'a+')
    # for j in cross_sections:
    #     cf.write(str(j)[1:-1] + ' ')
    #     cf.write('\n')
    # cf.close()



    '''重构'''
    R = []
    for Theta in arg.angle:
        t = np.pi * Theta / 180
        R.append(np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]))

    rotary_center = np.array(arg.center)

    '''可视化'''
    V1_tf = (Partial_V1 + np.array([arg.translation[0]]) - rotary_center.reshape(1, -1)) @ R[0]
    V2_tf = (Partial_V2 + np.array([arg.translation[1]]) - rotary_center.reshape(1, -1)) @ R[1]
    V3_tf = (Partial_V3 + np.array([arg.translation[2]]) - rotary_center.reshape(1, -1)) @ R[2]


    vis = o3d.visualization.Visualizer()  # 创建窗口对象
    vis.create_window(window_name='Result')  # 设置窗口标题

    visualization(V1_tf, vis, point_color=[1, 0, 0])  # 红色
    visualization(V2_tf, vis, point_color=[1, 1, 0])  # 黄色
    visualization(V3_tf, vis, point_color=[0, 0, 1])  # 蓝色

    vis.run()
    vis.destroy_window()




    Vs_tf = [V1_tf, V2_tf, V3_tf]
    # for i in range(arg.view_num):
    #     V_tf = Vs_tf[i]
    #     test_cache_file = result_path + f'/result_V{i+1}_tf.txt'
    #     cf = open(test_cache_file, 'a+')
    #     for j in V_tf:
    #         cf.write(str(j)[1:-1] + ' ')
    #         cf.write('\n')
    #     cf.close()




    '''可视化截面'''
    whole_profile = np.concatenate([V1_tf, V2_tf, V3_tf], axis=0)
    cross_sections = []
    for h in arg.height:
        Z_max = h + 0.13
        Z_min = h - 0.14
        index0 = whole_profile[:, 2] > Z_min
        index1 = whole_profile[:, 2] < Z_max

        S_h = whole_profile[index0 & index1]
        '''下采样'''
        num = S_h.shape[0]//1
        d = S_h.shape[0] // num
        d_index = [d * i for i in range(num)]
        cross_sections.append(S_h[d_index])

    cross_sections = np.concatenate(cross_sections, axis=0)
    # test_cache_file = result_path + f'/result_cross_sections_tf.txt'
    # cf = open(test_cache_file, 'a+')
    # for j in cross_sections:
    #     cf.write(str(j)[1:-1] + ' ')
    #     cf.write('\n')
    # cf.close()

    vis = o3d.visualization.Visualizer()  # 创建窗口对象
    vis.create_window(window_name='Cross-sections')  # 设置窗口标题

    visualization(cross_sections, vis, point_color=[1, 0, 0])  # 红色

    vis.run()
    vis.destroy_window()


    cross_sections = []
    for h in arg.height:
        Z_max = h + 0.03
        Z_min = h - 0.04
        index0 = whole_profile[:, 2] > Z_min
        index1 = whole_profile[:, 2] < Z_max

        S_h = whole_profile[index0 & index1]
        '''下采样'''
        num = S_h.shape[0]//6
        d = S_h.shape[0] // num
        d_index = [d * i for i in range(num)]
        cross_sections.append(S_h[d_index])


    # for i in range(len(arg.height)):
    #     cross_section = cross_sections[i]
    #     test_cache_file = result_path + f'/result_cross_section_{i+1}_tf.txt'
    #     cf = open(test_cache_file, 'a+')
    #     for j in cross_section:
    #         cf.write(str(j)[1:-1] + ' ')
    #         cf.write('\n')
    #     cf.close()




    cross_sections = []
    h = arg.height[0]
    for i, V_tf in enumerate(Vs_tf):
        Z_max = h + 0.03
        Z_min = h - 0.04
        index0 = V_tf[:, 2] > Z_min
        index1 = V_tf[:, 2] < Z_max

        S_h = V_tf[index0 & index1]

        # test_cache_file = result_path + f'/result_V{i+1}_cross_sections_1_tf.txt'
        # cf = open(test_cache_file, 'a+')
        # for j in S_h:
        #     cf.write(str(j)[1:-1] + ' ')
        #     cf.write('\n')
        # cf.close()

        '''下采样'''
        num = S_h.shape[0]//1
        d = S_h.shape[0] // num
        d_index = [d * i for i in range(num)]
        cross_sections.append(S_h[d_index])




