import numpy as np

def truncation(view, H = [42., 53.5], width = 5.):
    '''截取出所需截面'''
    cross_section = []
    for h in H:
        Z_max = h+width
        Z_min = h-width

        index0 = view[:, 2]>Z_min
        index1 = view[:, 2]<Z_max

        cross_section.append(view[index0 & index1])

    return np.concatenate(cross_section, axis=0)

def Sort(Data):
    # 排序
    List = []
    List.append(Data[0])
    center = Data[0]
    Data = Data[1:]
    while len(Data):

        distance = ((Data - center) ** 2).sum(-1)

        index = np.argmin(distance)
        # print(index)
        if distance[index] != 0.0:
            List.append(Data[index])

        center = Data[index]
        Data = np.delete(Data, index, 0)

    Data = np.stack(List)

    return Data