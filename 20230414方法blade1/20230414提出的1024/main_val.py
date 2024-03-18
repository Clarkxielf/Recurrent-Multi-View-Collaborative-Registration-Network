import time
import torch
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import scipy.io as sio

from utils_train_val.config import parse_args
from utils_train_val.network import net
from utils_train_val.loss import Loss
from utils_train_val.common import loaddata, stageprint, render_points_with_rgb, chamfer_dist, Alignment, adjust_learning_rate, visualization, set_seed


def update_val_cache(model, args):
    val_cache_file = result_path + '/result_cache.txt'
    error_meanpoints_list, error_stdpoints_list, error_meancenter_list, error_stdcenter_list, time_avg_list = compute_val_loss_values(model, args)
    cf = open(val_cache_file, 'a+')
    for j in [error_meanpoints_list, error_stdpoints_list, error_meancenter_list, error_stdcenter_list, time_avg_list]:
        for i in j:
            cf.write(str(i) + ' ')
        cf.write('\n')
    cf.close()

def compute_val_loss_values(model, args):
    error_meanpoints_list = []
    error_stdpoints_list = []
    error_meancenter_list = []
    error_stdcenter_list = []
    all_modified_rotary_center = {}
    batch_cnt = 0.0
    time_cnt = 0.0
    print('Computing the validation loss on the validation set:')
    for s in range(0, 2 * val_dataset_num, 1):  # 测试10个批次
        start_pos = s * args.batchsize
        end_pos = start_pos + args.batchsize

        if end_pos > val_dataset_num:
            end_pos = val_dataset_num

        this_batch_size = end_pos - start_pos

        dispoints_list, discenter_list, elapsed, modified_rotary_center_list = val_one_batch(model, start_pos, end_pos, args)

        if s==0:
            dispointslist = dispoints_list
            discenterlist = discenter_list
            for i in range(args.iteration):
                all_modified_rotary_center[str(i)] = modified_rotary_center_list[i].detach().cpu().numpy()
        else:
            for i in range(args.iteration):
                dispointslist[i] = np.concatenate([dispointslist[i], dispoints_list[i]], axis=0)
                discenterlist[i] = np.concatenate([discenterlist[i], discenter_list[i]], axis=0)

                all_modified_rotary_center[str(i)] = np.concatenate([all_modified_rotary_center[str(i)], modified_rotary_center_list[i].detach().cpu().numpy()], axis=0)

        batch_cnt += this_batch_size
        time_cnt += elapsed

        if end_pos == val_dataset_num:
            break

    print(f'val_dataset_num: {val_dataset_num}')
    print(f'batch_cnt: {batch_cnt}')
    time_avg = time_cnt / batch_cnt / (args.view_num - 1)
    print(f'time_avg: {time_avg}')

    for i in range(args.iteration):
        error_meanpoints_list.append(np.mean(dispointslist[i]))
        error_stdpoints_list.append(np.std(dispointslist[i]))
        error_meancenter_list.append(np.mean(discenterlist[i]))
        error_stdcenter_list.append(np.std(discenterlist[i]))

    all_modified_rotary_center_file = result_path + '/result_all_modified_rotary_center.mat'
    sio.savemat(all_modified_rotary_center_file, all_modified_rotary_center)

    return error_meanpoints_list, error_stdpoints_list, error_meancenter_list, error_stdcenter_list, [time_avg]


def val_one_batch(model, start_pos, end_pos, args):
    val_Viewpoints_batch = val_Viewpoints_tensor[start_pos:end_pos].cuda()  # batch*view_num*point_num*3
    val_R_batch = val_R_tensor[start_pos:end_pos].cuda()  # batch*view_num*3*3
    val_T_batch = val_T_tensor[start_pos:end_pos].cuda()  # batch*view_num*3
    val_Rotary_Center_batch = val_Rotary_Center_tensor[start_pos:end_pos].cuda()  # batch*3
    R_init_batch = R_init[start_pos:end_pos]  # batch*3*3
    offset_Q_batch = offset_Q[start_pos:end_pos]  # batch*3

    color = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]]  # 红、黄、蓝、绿

    # model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list = model(val_Viewpoints_batch, val_R_batch, val_T_batch, args.view_num)
        elapsed = (time.perf_counter() - start)
        print(f"Time used:{elapsed}")

        print(modified_rotary_center_list)
        print(val_Rotary_Center_batch)

        iter_time = len(weight_list)
        for i in range(iter_time):
            for j in range(args.view_num):
                align_viewpoints_list[i][j] = align_viewpoints_list[i][j].squeeze().transpose(-1, -2).detach().cpu().numpy() # batch*3*point_num--->point_num*3


        '''还原视场'''
        Viewpoints = val_Viewpoints_batch[0].cpu().numpy()
        center = val_Rotary_Center_batch[0].cpu().numpy()
        R = val_R_batch[0].cpu().numpy()
        T = val_T_batch[0].cpu().numpy()
        for k in range(args.view_num):
            Viewpoints[k] = (Viewpoints[k] + T[k][None, ...]- center[None, ...]) @ R[k]
        viewpoints_target = Viewpoints.reshape(-1, 3)

        dispoints_list = []
        discenter_list = []
        for i in range(iter_time):# 所有阶
            viewpoints_pred = np.concatenate(align_viewpoints_list[i], axis=0)
            dispoints = ((((viewpoints_pred[:, None, :].repeat(viewpoints_target.shape[0], 1) - viewpoints_target[None, ...])**2).sum(-1)).min(-1))**0.5
            dispoints_list.append(dispoints)
            discenter = ((((val_Rotary_Center_batch - modified_rotary_center_list[i]) ** 2).sum(-1)) ** 0.5).detach().cpu().numpy()
            discenter_list.append(discenter)

            print(f'stage {i} validation center chamfer error {((((val_Rotary_Center_batch - modified_rotary_center_list[i]) ** 2).sum(-1)) ** 0.5).item()}')

    return dispoints_list, discenter_list, elapsed, modified_rotary_center_list


if __name__=='__main__':
    set_seed(SEED=2023)

    args = parse_args()
    print(f'args:{args}')

    val_datasets_path = os.getcwd().split('/20230414方法blade1')[0] + '/' + args.train_val_root_path + '/' + args.val_datasets_mat_path
    print('The val_datasets_mat_file path is: ', val_datasets_path)

    # batch*view_num*point_num*3；batch*view_num*3*3；batch*view_num*3；batch*3；batch*3*3；batch*3
    val_Viewpoints_tensor, val_R_tensor, val_T_tensor, val_Rotary_Center_tensor, R_init, offset_Q = loaddata(val_datasets_path, args.point_num)

    val_dataset_num = val_Viewpoints_tensor.shape[0]
    print(f'val_dataset_num: {val_dataset_num}')

    result_path = './results_' + str(args.point_num) + '_' + args.val_out_dir
    if os.path.exists(result_path):
        os.system('rm -rf ' + result_path)
    os.makedirs(result_path)

    rc_net = net().cuda()
    rc_net.load_state_dict(torch.load(f'./results_{args.point_num}_training_results_sample/sample_{args.last_sample_id}.pt'), True)

    update_val_cache(rc_net, args)
