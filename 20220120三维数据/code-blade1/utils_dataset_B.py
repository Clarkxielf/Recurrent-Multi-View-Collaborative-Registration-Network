import numpy as np

def data3D(view):
    '''将.csv文件转换为M*3数据'''
    X = view[0]
    Z = view[:, 0]
    index_X = ~np.isnan(X)
    index_Z = ~np.isnan(Z)

    new_index_Z = []
    for j,_ in enumerate(index_Z):
        if j%2==1:
            new_index_Z.append(j)

    X = X[index_X]
    Z = Z[new_index_Z]

    data = view[new_index_Z][:, index_X]
    XYZ = []
    for i, data_Z in enumerate(Z):
        Y = data[i]
        index_Y = ~np.isnan(Y)
        data_Y = Y[index_Y]
        data_X = X[index_Y]
        data_Z = np.repeat(data_Z, len(data_X))
        xyz = np.stack([data_X, data_Y, data_Z])
        XYZ.append(xyz)

    XYZ = np.concatenate(XYZ, axis=-1).transpose([1, 0])  # N*3

    return XYZ

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

def farthest_subsample_points(points, npoint):
    '''
    最远点采样, 返回采样点
    points: N*3,Numpy
    npoint: 采样点个数,int
    '''

    N, C = points.shape  # N,3
    centroids = np.zeros(npoint, dtype=int)  # 1024,用于记录选取的1024个点的索引
    distance = np.ones(N) * 1e20  # 1024,用于记录1024个全部数据点与已采样点的距离
    farthest = np.random.randint(0, N, dtype=int)  # 第一个点从0~N中随机选取
    for i in range(npoint):
        centroids[i] = farthest  # 第一个点随机选取
        centroid = points[farthest, :].reshape(1, 3)  # 获取当前采样点的坐标,(x,y,z)
        dist = np.sum((points - centroid) ** 2, -1)  # 计算1024个全部采样点与当前采样点的欧式距离
        mask = dist < distance  # 为更新每个点到已采样点的距离做标记
        distance[mask] = dist[mask]  # 更新每个点到已采样点的距离
        farthest = np.argmax(distance)  # 选取到已采样点距离最大的点作为下一个采样点

        print(f'Sampling in progress: {i}/{N}')

    Sampling_Points = points[centroids, :]

    return Sampling_Points  # npoint*3