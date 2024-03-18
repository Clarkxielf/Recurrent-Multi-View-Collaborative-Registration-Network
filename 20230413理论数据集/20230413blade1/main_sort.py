import pandas as pd
import argparse
import numpy as np
import scipy.io as sio

from utils_sort import truncation, Sort

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('-p', '--path', help='Path to data files', required=False, default='.')
    parser.add_argument('-H', '--height', help='Height of cross-section', required=False, default=[42., 53.5, 79.])
    arg = parser.parse_args()

    # '''加载数据/处理数据'''
    # Cross_sections = pd.read_table(arg.path+'/20230413-a142a-201003a002----生成点云数据.asc', header=None).values
    # List = []
    # for i, d in enumerate(Cross_sections):
    #     x, y, z = d[0].split(' ')
    #     x, y, z = float(x), float(y), float(z)
    #     List.append(np.stack([x, y, z]))
    #     print(f'{i+1}/{Cross_sections.shape[0]}')
    # Cross_sections = np.stack(List)
    # sio.savemat('./Cross_sections.mat', {'Cross_sections': Cross_sections})



    '''截取特定截面,此处无需调整Z轴'''
    Cross_sections = sio.loadmat('./Cross_sections.mat')['Cross_sections']

    width = 1.
    Cross_sections = truncation(Cross_sections, H=arg.height, width=width)

    '''分层处理,截面排序'''
    z_max, z_min = max(Cross_sections[:, 2]), min(Cross_sections[:, 2])
    len_z_axis = len(list(np.arange(start=z_min, stop=z_max+0.01, step=0.06)))

    List = []
    num_section = 0
    for j in range(len_z_axis):
        Condition1 = (Cross_sections[:, 2]-0.06*j-z_min)>(-0.04)
        Condition2 = (Cross_sections[:, 2]-0.06*j-z_min)<(+0.03)
        section = Cross_sections[Condition1 & Condition2]
        if section.shape[0]!=0:
            print(f'Sorting in progress: {num_section+1}/{len_z_axis}')

            section = Sort(section) # 按距离排序
            List.append(section)

            num_section += 1

    Cross_sections = np.concatenate(List, axis=0)

    sio.savemat(arg.path+'/sort_Cross_sections.mat', {'Cross_sections': Cross_sections})