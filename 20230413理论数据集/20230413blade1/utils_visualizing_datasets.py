import open3d as o3d


def visualization(view, vis, point_color=[1, 1, 1]):

    pcd = o3d.open3d.geometry.PointCloud() # 创建点云对象
    pcd.points = o3d.open3d.utility.Vector3dVector(view) # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.paint_uniform_color(point_color) # 设置点的颜色
    vis.add_geometry(pcd) # 将点云加入到窗口中