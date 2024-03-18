import argparse
import numpy as np
import scipy.io as sio

from utils_dataset_B import farthest_subsample_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('-p', '--path', help='Path to data files', required=False, default='.')
    parser.add_argument('-n', '--num_view', help='Number of viewpoint', required=False, default=3)
    parser.add_argument('-d', '--dataset', help='Dataset types', required=False, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('-s', '--size', help='Number of Samples', required=False, default=2)
    parser.add_argument('-N', '--npoints', help='Number of downsampling points', required=False, default=2048)

    parser.add_argument('-c', '--center', help='Rotary_Center', required=False, default=[3.0, -5.0, 0.])
    parser.add_argument('-o', '--offset', help='offset Q', required=False, default=[-1.0,  1.0,  0.        ])
    parser.add_argument('-a', '--angle', help='Motion Parameters (Angle)', required=False, default=[0, 60, 160])
    # parser.add_argument('-t', '--translation', help='Motion Parameters', required=False, default=[[0, 0, 0], [0, 0, 0], [5, -5, 0]])
    parser.add_argument('-H', '--height', help='Height of Cross-section', required=False,
                        default=[42.00, 53.5, 79.00], choices=[[42., 53.5, 79.], []])
    arg = parser.parse_args()

    label_R_init = []
    label_center = []
    label_offset_Q = []
    label_R = []
    label_T = []
    label_viewpoints = []
    for s in range(arg.size):
        print(f'Samples are being made: {s + 1}/{arg.size}')

        '''先将模型放在质心'''
        Cross_sections = sio.loadmat(arg.path + '/sort_Cross_sections.mat')['Cross_sections']
        Cross_sections[:, 0] = -1 * Cross_sections[:, 0]  # 坐标朝向与测量数据保持一致

        Q = np.mean(Cross_sections, axis=0)
        Q[-1] = 0.
        Cross_sections = Cross_sections - Q

        '''随机初始化模型位姿'''
        random_init_theta = np.pi * 140 / 180 + (15 * np.pi / 180) * 2 * (np.random.rand(1).item() - 0.5)  # 角度：[-15, 15]
        R_init = np.array([[np.cos(random_init_theta), -np.sin(random_init_theta), 0], [np.sin(random_init_theta), np.cos(random_init_theta), 0], [0, 0, 1]])
        Cross_sections = Cross_sections @ R_init
        label_R_init.append(R_init)

        '''生成num_view个视场'''
        '''随机化回转中心'''
        center = 10 * 2 * np.concatenate([np.random.rand(2) - 0.5, np.array([0])])  # 回转中心：[-10, 10]
        label_center.append(center) # S 3
        '''随机化质心偏移'''
        offset_Q = 2 * 2 * np.concatenate([np.random.rand(2) - 0.5, np.array([0])])  # 质心偏移：[-2, 2]
        Cross_sections += offset_Q
        label_offset_Q.append(offset_Q)

        Viewpoints = {}
        R = []
        T = []
        for i in range(arg.num_view):
            '''激光器在第 i 个采集工位下叶片的位姿'''
            theta = np.pi * arg.angle[i] / 180 + (2 * np.pi / 180) * 2 * (np.random.random(1).item() - 0.5)  # 每个工位角度：[-2, 2]
            Ri = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            R.append(Ri)

            Viewpoints[f'V{i + 1}'] = Cross_sections @ Ri.T

            Ti = 0.2 * 2 * np.concatenate([np.random.rand(2) - 0.5, np.array([0])])  # 每个工位平移：[-0.2, 0.2]
            T.append(Ti)
            Viewpoints[f'V{i + 1}'] = Viewpoints[f'V{i + 1}'] + center - Ti

        label_R.append(np.stack(R, axis=0))  # S num_view*3*3
        label_T.append(np.stack(T, axis=0))  # S num_view*3

        for i in range(arg.num_view):
            '''分割视场'''
            z_max, z_min = max(Viewpoints[f'V{i + 1}'][:, 2]), min(Viewpoints[f'V{i + 1}'][:, 2])
            len_z_axis = len(list(np.arange(start=z_min, stop=z_max + 0.01, step=0.06)))

            List = []
            num_section = 0
            for j in range(len_z_axis):
                Condition1 = (Viewpoints[f'V{i + 1}'][:, 2] - 0.06 * j - z_min) >= (-0.04)
                Condition2 = (Viewpoints[f'V{i + 1}'][:, 2] - 0.06 * j - z_min) < (+0.03)
                section = Viewpoints[f'V{i + 1}'][Condition1 & Condition2]
                if section.shape[0] != 0:
                    # print(f'Segmentation in process: {num_section + 1}/{len_z_axis}')
                    num_section += 1

                    index_x_min = np.argmin(section[:, 0])
                    section = np.concatenate([section[index_x_min:, :], section[:index_x_min, :]], axis=0)  # x最小值放在第一个

                    index_x_max = np.argmax(section[:, 0])
                    if section[0, 1] < section[0 + 1, 1]:
                        section = section[:index_x_max + 1, :]  # 顺时针
                    else:
                        section = np.concatenate([section[index_x_max:, :], section[0, :][None, :]], axis=0)  # 逆时针
                        section = section[::-1]  # 逆序，x从小到大

                    '''去除遮挡视场'''
                    index_y_max = np.argmax(section[:, 1])
                    if section[index_y_max, 0] > section[(index_y_max + len(section)) // 2, 0]:
                        section = section[:index_y_max]

                    index_y_max = np.argmax(section[:, 1])
                    if section[index_y_max, 0] < section[(index_y_max) // 2, 0]:
                        section = section[index_y_max:]

                    List.append(section)

            print(f'Number of processed cross-sections in viewpoint {i + 1}: {num_section}')
            # for i in range(len(List)):
            #     plt.scatter(List[i][:, 0], List[i][:, 1])
            #     plt.show()

            Viewpoints[f'V{i + 1}'] = np.concatenate(List, axis=0)

        for i in range(arg.num_view):
            '''等dx=0.02采样'''
            x_max, x_min = max(Viewpoints[f'V{i + 1}'][:, 0]), min(Viewpoints[f'V{i + 1}'][:, 0])
            y_max, y_min = max(Viewpoints[f'V{i + 1}'][:, 1]), min(Viewpoints[f'V{i + 1}'][:, 1])
            z_max, z_min = max(Viewpoints[f'V{i + 1}'][:, 2]), min(Viewpoints[f'V{i + 1}'][:, 2])

            x_axis = list(np.arange(start=x_min, stop=x_max + 0.01, step=0.02))
            len_z_axis = len(list(np.arange(start=z_min, stop=z_max + 0.01, step=0.06)))

            List = []
            num_section = 0
            for j in range(len_z_axis):
                '''分层处理'''
                Condition1 = (Viewpoints[f'V{i + 1}'][:, 2] - 0.06 * j - z_min) >= (-0.04)
                Condition2 = (Viewpoints[f'V{i + 1}'][:, 2] - 0.06 * j - z_min) < (+0.03)
                section = Viewpoints[f'V{i + 1}'][Condition1 & Condition2]

                if section.shape[0] != 0:
                    print(f'Segmentation in process: {num_section + 1}/{len_z_axis}')
                    num_section += 1
                    '''寻找离轴最近的点'''
                    index_x = []
                    for k in x_axis:
                        idx = np.argmin(np.abs(section[:, 0] - k))
                        if idx not in index_x:
                            index_x.append(idx)

                    index_x = np.sort(index_x)
                    section = section[index_x]
                    '''已采样的点'''
                    List.append(section)

            print(f'Number of cross-sections processed in viewpoint {i + 1}: {num_section}')
            Viewpoints[f'V{i + 1}'] = np.concatenate(List, axis=0)

        for i in range(arg.num_view):
            '''传感器视场范围：安装净距离CD:47.5mm；视野FOV:25-32.5mm；测量范围MR:25mm'''
            z_max, z_min = max(Viewpoints[f'V{i + 1}'][:, 2]), min(Viewpoints[f'V{i + 1}'][:, 2])
            len_z_axis = len(list(np.arange(start=z_min, stop=z_max + 0.01, step=0.06)))

            List = []
            num_section = 0
            for j in range(len_z_axis):
                '''分层处理'''
                Condition1 = (Viewpoints[f'V{i + 1}'][:, 2] - 0.06 * j - z_min) >= (-0.04)
                Condition2 = (Viewpoints[f'V{i + 1}'][:, 2] - 0.06 * j - z_min) < (+0.03)
                section = Viewpoints[f'V{i + 1}'][Condition1 & Condition2]

                if section.shape[0] != 0:
                    print(f'Segmentation in process: {num_section + 1}/{len_z_axis}')
                    num_section += 1
                    '''测量范围MR:25mm'''
                    MR = 25
                    index_min_y = np.argmin(section[:, 1])
                    section = np.concatenate([section[index_min_y][None, ...], np.delete(section, index_min_y, 0)],
                                             axis=0)

                    index_y = []
                    index_y.append(0)
                    for k in range(1, section.shape[0]):
                        if abs(section[k, 1] - section[0, 1]) < MR:
                            index_y.append(k)
                        else:
                            continue

                    section = section[index_y]
                    '''视野FOV:28mm'''
                    FOV = 28
                    index_min_x = np.argmin(section[:, 0])
                    section = np.concatenate([section[index_min_x][None, ...], np.delete(section, index_min_x, 0)],
                                             axis=0)

                    index_x = []
                    index_x.append(0)
                    for k in range(1, section.shape[0]):
                        if abs(section[k, 0] - section[0, 0]) < FOV:
                            index_x.append(k)
                        else:
                            continue

                    index_x = np.sort(index_x)
                    section = section[index_x]
                    '''符合传感器视场范围的点'''
                    List.append(section)

            print(f'Number of cross-sections processed in viewpoint {i + 1}: {num_section}')
            Viewpoints[f'V{i + 1}'] = np.concatenate(List, axis=0)

        viewpoints = []
        for i in range(arg.num_view):
            '''下采样到固定点数'''
            Viewpoints[f'V{i + 1}'] = farthest_subsample_points(Viewpoints[f'V{i + 1}'], arg.npoints)
            viewpoints.append(Viewpoints[f'V{i + 1}'])

        label_viewpoints.append(np.stack(viewpoints, axis=0))  # S num_view*npoints*3

    label_R_init = np.stack(label_R_init, axis=0)  # S*3*3
    label_center = np.stack(label_center, axis=0)  # S*3
    label_offset_Q = np.stack(label_offset_Q, axis=0) # S*3
    label_R = np.stack(label_R, axis=0)  # S*num_view*3*3
    label_T = np.stack(label_T, axis=0)  # S*num_view*3
    label_viewpoints = np.stack(label_viewpoints, axis=0)  # S*num_view*npoints*3

    sio.savemat(arg.path + f'/dataset_{arg.dataset}_{arg.size}_{arg.npoints}_45.mat',
                {'viewpoints': label_viewpoints, 'R': label_R, 'T':label_T, 'Rotary_Center': label_center, 'R_init': label_R_init, 'offset_Q': label_offset_Q})
