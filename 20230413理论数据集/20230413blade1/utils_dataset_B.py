import numpy as np

def farthest_subsample_points(points, npoint):
    '''
    最远点采样, 返回采样点
    points: N*3,Numpy
    npoint: 采样点个数,int
    '''

    N, C = points.shape #N,3
    centroids = np.zeros(npoint, dtype=int) # 1024,用于记录选取的1024个点的索引
    distance = np.ones(N)*1e20 # 1024,用于记录1024个全部数据点与已采样点的距离
    farthest = np.random.randint(0, N, dtype=int) # 第一个点从0~N中随机选取
    for i in range(npoint):
        centroids[i] = farthest # 第一个点随机选取
        centroid = points[farthest, :].reshape(1, 3) # 获取当前采样点的坐标,(x,y,z)
        dist = np.sum((points-centroid)**2, -1) # 计算1024个全部采样点与当前采样点的欧式距离
        mask = dist<distance # 为更新每个点到已采样点的距离做标记
        distance[mask] = dist[mask] # 更新每个点到已采样点的距离
        farthest = np.argmax(distance) # 选取到已采样点距离最大的点作为下一个采样点

        print(f'Sampling in progress: {i}/{N}')

    Sampling_Points = points[centroids, :]

    return Sampling_Points # npoint*3