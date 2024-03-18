## 处理数模
+ 1.将数模文件“\*\*\*.igs”离散为点云数据，其点间距设置为：d=0.01mm, dz=0.06mm；并且将该点云数据保存到“\*\*\*.asc”文件
## 代码说明
+ 1.main_sort.py: 对每个截面上的点进行排序，并保存到“\*\*\*.mat”文件
+ 2.main_dataset_B.py: 生成理论数据集
+ 3.main_visualizing_datasets.py: 可视化检验样本正确性
+ 4.main_dataset_B_for_point_cloud_registration.py: 将本文数据集变换为点云配准数据集
+ 5.main_visualizing_datasets_for_point_cloud_registration.py: 验证点云配准数据集正确性
+ 6.main_noise.py: 对测试集数据增加噪声
+ 7.main_noise_for_point_cloud_registration.py: 将本文噪声数据集变换为点云配准数据集