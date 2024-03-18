import torch
import glob
import numpy as np
import scipy.io as sio
import open3d as o3d

def set_seed(SEED=2023):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED=2023)

def loaddata(paths, point_num):
    paths = glob.glob(paths)
    paths.sort()

    for i, path in enumerate(paths):
        if i == 0:
            Viewpoints = sio.loadmat(path)['viewpoints'].astype(np.float32)  # batch*view_num*point_num*3
            R = sio.loadmat(path)['R'].astype(np.float32)  # batch*view_num*3*3
            T = sio.loadmat(path)['T'].astype(np.float32)  # batch*view_num*3
            Rotary_Center = sio.loadmat(path)['Rotary_Center'].astype(np.float32)  # batch*3
            R_init = sio.loadmat(path)['R_init'].astype(np.float32)  # batch*3*3
            offset_Q = sio.loadmat(path)['offset_Q'].astype(np.float32)  # batch*3
        else:
            Viewpoints = np.concatenate([Viewpoints, sio.loadmat(path)['viewpoints'].astype(np.float32)], axis=0)
            R = np.concatenate([R, sio.loadmat(path)['R'].astype(np.float32)], axis=0)
            T = np.concatenate([T, sio.loadmat(path)['T'].astype(np.float32)], axis=0)
            Rotary_Center = np.concatenate([Rotary_Center, sio.loadmat(path)['Rotary_Center'].astype(np.float32)], axis=0)
            R_init = np.concatenate([R_init, sio.loadmat(path)['R_init'].astype(np.float32)], axis=0)
            offset_Q = np.concatenate([offset_Q, sio.loadmat(path)['offset_Q'].astype(np.float32)], axis=0)

    Viewpoints_tensor = torch.from_numpy(Viewpoints[:, :, :point_num, :])
    R_tensor = torch.from_numpy(R)
    T_tensor = torch.from_numpy(T)
    Rotary_Center_tensor = torch.from_numpy(Rotary_Center)

    return Viewpoints_tensor, R_tensor, T_tensor, Rotary_Center_tensor, R_init, offset_Q

def Alignment(center, Viewpoints, R, T, view_num):

    new_Viewpoints = []
    for i in range(view_num):
        new_Viewpoints.append(((Viewpoints[i].transpose(1, 2) + T[i].unsqueeze(1)- center.unsqueeze(1)) @ R[i]).transpose(1, 2))

    return new_Viewpoints

def render_points_with_rgb(points, rgb, savepath):
    f = open(savepath, 'w')
    for j in range(len(points)):
        point = points[j].transpose(-1, -2).cpu().numpy().reshape(-1, 3)

        for i in range(point.shape[0]):
            f.write('v '+str(point[i][0])+' '+str(point[i][1])+' '+str(point[i][2])+' '+str(rgb['r'][j])+' '+str(rgb['g'][j])+' '+str(rgb['b'][j])+'\n')
    f.close()


def chamfer_dist(points_x, points_y):
    '''
    points_x : batchsize * M * 3
    points_y : batchsize * N * 3
    output   : batchsize * M, batchsize * N
    '''
    thisbatchsize = points_x.size()[0]
    sqrdis = compute_sqrdis_map(points_x, points_y) # batch*point_num*point_num
    dist1 = sqrdis.min(dim=2)[0].view(thisbatchsize, -1) # batch*point_num
    dist2 = sqrdis.min(dim=1)[0].view(thisbatchsize, -1)
    return dist1**0.5, dist2**0.5

def compute_sqrdis_map(points_x, points_y):
    '''
    points_x : batchsize * M * 3
    points_y : batchsize * N * 3
    output   : batchsize * M * N
    '''
    pn_y = points_y.size()[1]

    sqrdis = ((points_x.unsqueeze(2).repeat(1, 1, pn_y, 1)-points_y.unsqueeze(1))**2).sum(-1)

    return sqrdis

def stageprint(vlist, text):
    for i in range(len(vlist)):
        print('stage ' + str(i) + ' ' + text + ' : ' + str(vlist[i]))

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*0.5 # 0.5**(1/n) 学习率每n代减半

def RandomJitter(viewpoint, scale=0.01, clip=0.05): # viewpoint: batch*3*point_num
    """ generate perturbations """

    noise = np.clip(np.random.normal(0.0, scale=scale, size=(viewpoint.shape[0], 2, viewpoint.shape[-1])), a_min=-clip, a_max=clip)
    viewpoint[:, :2, :] += torch.from_numpy(noise).cuda()  # Add noise to xy

    return viewpoint


def visualization(view, vis, point_color=[1, 1, 1]):

    pcd = o3d.open3d.geometry.PointCloud() # 创建点云对象
    pcd.points = o3d.open3d.utility.Vector3dVector(view) # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.paint_uniform_color(point_color) # 设置点的颜色
    vis.add_geometry(pcd) # 将点云加入到窗口中


def visualization2(view, vis, point_color=[1, 1, 1]):
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

    pcd = o3d.open3d.geometry.PointCloud() # 创建点云对象
    pcd.points = o3d.open3d.utility.Vector3dVector(XYZ) # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.paint_uniform_color(point_color) # 设置点的颜色
    vis.add_geometry(pcd)     # 将点云加入到窗口中

    return XYZ

def truncation(view, Z_max = +0.05, Z_min = -0.04):
    '''截取出所需截面'''
    index0 = view[:, 2]>Z_min
    index1 = view[:, 2]<Z_max

    return view[index0 & index1]