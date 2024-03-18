import glob
import torch
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import scipy.io as sio

from utils_train_val.config import parse_args
from utils_train_val.network import net
from utils_train_val.loss import Loss
from utils_train_val.common import stageprint, render_points_with_rgb, chamfer_dist, Alignment, adjust_learning_rate, visualization, set_seed


def update_inference_cache(model, args):
    compute_inference_loss_values(model, args)


def compute_inference_loss_values(model, args):
    batch_cnt = 0.0
    print('Computing the inference loss on the inference set:')
    for s in range(0, 2*inference_dataset_num, 1): # 测试10个批次
        start_pos = s*args.batchsize
        end_pos = start_pos+args.batchsize

        if end_pos > inference_dataset_num:
            end_pos = inference_dataset_num

        this_batch_size = end_pos-start_pos
        inference_one_batch(model, start_pos, end_pos, args)


        batch_cnt += this_batch_size

        if end_pos==inference_dataset_num:
            break

    print(f'inference_dataset_num: {inference_dataset_num}')
    print(f'batch_cnt: {batch_cnt}')


def inference_one_batch(model, start_pos, end_pos, args):

    inference_Viewpoints_batch = inference_Viewpoints_tensor[start_pos:end_pos].cuda() # batch*view_num*point_num*3
    inference_R_batch = inference_R_tensor[start_pos:end_pos].cuda() # batch*view_num*3*3
    inference_T_batch = inference_T_tensor[start_pos:end_pos].cuda() # batch*view_num*3
    inference_Rotary_Center_batch = inference_Rotary_Center_tensor[start_pos:end_pos].cuda() # batch*3

    color = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]]  # 红、黄、蓝、绿

    # model.eval()
    with torch.no_grad():
        weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list = model(inference_Viewpoints_batch, inference_R_batch, inference_T_batch, args.view_num)

        print(modified_rotary_center_list)
        print(inference_Rotary_Center_batch)


def loaddata(paths, point_num):
    paths = glob.glob(paths)
    paths.sort()

    for i, path in enumerate(paths):
        if i == 0:
            Viewpoints = sio.loadmat(path)['viewpoints'].astype(np.float32)  # batch*view_num*point_num*3
            R = sio.loadmat(path)['R'].astype(np.float32)  # batch*view_num*3*3
            N = R.shape[0]
            T = np.zeros((N, args.view_num, 3)).astype(np.float32)  # batch*view_num*3
            Rotary_Center = sio.loadmat(path)['Rotary_Center'].astype(np.float32)  # batch*3
        else:
            Viewpoints = np.concatenate([Viewpoints, sio.loadmat(path)['viewpoints'].astype(np.float32)], axis=0)
            R = np.concatenate([R, sio.loadmat(path)['R'].astype(np.float32)], axis=0)
            T = np.concatenate([T, np.zeros((N, args.view_num, 3)).astype(np.float32)], axis=0)
            Rotary_Center = np.concatenate([Rotary_Center, sio.loadmat(path)['Rotary_Center'].astype(np.float32)], axis=0)

    Viewpoints_tensor = torch.from_numpy(Viewpoints[:, :, :point_num, :])
    R_tensor = torch.from_numpy(R)
    T_tensor = torch.from_numpy(T)
    Rotary_Center_tensor = torch.from_numpy(Rotary_Center)

    return Viewpoints_tensor, R_tensor, T_tensor, Rotary_Center_tensor


if __name__=='__main__':
    set_seed(SEED=2023)

    args = parse_args()
    print(f'args:{args}')

    inference_datasets_path = os.getcwd().split('/20230414方法blade1')[0] + '/' + args.inference_root_path + '/' + args.inference_datasets_mat_path
    print('The inference_datasets_mat_file path is: ', inference_datasets_path)

    # batch*view_num*point_num*3；batch*view_num*3*3；batch*view_num*3；batch*3；batch*3*3；batch*3
    inference_Viewpoints_tensor, inference_R_tensor, inference_T_tensor, inference_Rotary_Center_tensor = loaddata(inference_datasets_path, args.point_num)

    inference_dataset_num = inference_Viewpoints_tensor.shape[0]
    print(f'inference_dataset_num: {inference_dataset_num}')

    result_path = './results_' + str(args.point_num) + '_' + args.inference_out_dir
    if os.path.exists(result_path):
        os.system('rm -rf ' + result_path)
    os.makedirs(result_path)

    rc_net = net().cuda()
    rc_net.load_state_dict(torch.load(f'./results_{args.point_num}_training_results_sample/sample_{args.last_sample_id}.pt'), True)

    update_inference_cache(rc_net, args)
