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



def update_test_cache(args):
    test_cache_file = result_path + '/result_cache.txt'
    error_meanpoints_cs_list, error_stdpoints_cs_list = compute_test_loss_values(args)
    cf = open(test_cache_file, 'a+')
    for j in [error_meanpoints_cs_list, error_stdpoints_cs_list]:
        for i in j:
            cf.write(str(i) + ' ')
        cf.write('\n')
    cf.close()

def compute_test_loss_values(args):
    error_meanpoints_cs_list = []
    error_stdpoints_cs_list = []
    batch_cnt = 0.0
    print('Computing the testing loss on the testing set:')
    for s in range(0, 2*test_dataset_num, 1): # 测试10个批次
        start_pos = s*args.batchsize
        end_pos = start_pos+args.batchsize

        if end_pos > test_dataset_num:
            end_pos = test_dataset_num

        this_batch_size = end_pos-start_pos

        dispoints_cs_list = test_one_batch(start_pos, end_pos, args)

        if s==0:
            dispointscs_list = dispoints_cs_list
        else:
            for i in range(args.iteration):
                for j in range(len(args.height)):
                    dispointscs_list[i][j] = np.concatenate([dispointscs_list[i][j], dispoints_cs_list[i][j]], axis=0)

        batch_cnt += this_batch_size

        if end_pos==test_dataset_num:
            break

    print(f'test_dataset_num: {test_dataset_num}')
    print(f'batch_cnt: {batch_cnt}')


    for i in range(args.iteration):

        error_meanpoints_cs_list_onece = []
        error_stdpoints_cs_list_onece = []
        for j in range(len(args.height)):
            error_meanpoints_cs_list_onece.append(np.mean(dispointscs_list[i][j]))
            error_stdpoints_cs_list_onece.append(np.std(dispointscs_list[i][j]))

        error_meanpoints_cs_list.append(error_meanpoints_cs_list_onece)
        error_stdpoints_cs_list.append(error_stdpoints_cs_list_onece)

    return error_meanpoints_cs_list, error_stdpoints_cs_list


def test_one_batch(start_pos, end_pos, args):

    test_Viewpoints_batch = test_Viewpoints_tensor[start_pos:end_pos].cuda() # batch*view_num*point_num*3
    test_R_batch = test_R_tensor[start_pos:end_pos].cuda() # batch*view_num*3*3
    test_T_batch = test_T_tensor[start_pos:end_pos].cuda() # batch*view_num*3
    test_Rotary_Center_batch = test_Rotary_Center_tensor[start_pos:end_pos].cuda() # batch*3
    R_init_batch = R_init[start_pos:end_pos] # batch*3*3
    offset_Q_batch = offset_Q[start_pos:end_pos] # batch*3

    color = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]]  # 红、黄、蓝、绿

    # model.eval()
    with torch.no_grad():
        Viewpoints_list = []
        R_list = []
        T_list = []
        for i in range(args.view_num):
            if args.noise_type=='clean':
                Viewpoints_list.append(test_Viewpoints_batch[:, i, :args.point_num_val2, :].transpose(1, 2)) # view_num batch*3*point_num
            else:
                raise NotImplementedError

            R_list.append(test_R_batch[:, i, :, :])  # view_num batch*3*3
            T_list.append(test_T_batch[:, i, :])  # view_num batch*3

        align_viewpoints_list = []
        modified_rotary_center_list = []
        for i in range(args.iteration):
            modified_OR = torch.from_numpy(all_modified_rotary_center_dict[str(i)][start_pos:end_pos]).cuda() # batch*3
            modified_rotary_center_list.append(modified_OR)

            # batch*3, view_num batch*3*point_num, view_num batch*3*3, view_num batch*3---> view_num batch*3*point_num
            align_viewpoints = Alignment(modified_OR, Viewpoints_list, R_list, T_list, args.view_num)
            align_viewpoints_list.append(align_viewpoints) # [[*, *, *], ..., [*, *, *]]

        for i in range(args.iteration):
            for j in range(args.view_num):
                align_viewpoints_list[i][j] = align_viewpoints_list[i][j].squeeze().transpose(-1, -2).detach().cpu().numpy() # batch*3*point_num--->point_num*3

        '''还原视场'''
        Viewpoints = test_Viewpoints_batch[0, :, :args.point_num_val2, :].cpu().numpy()
        center = test_Rotary_Center_batch[0].cpu().numpy()
        R = test_R_batch[0].cpu().numpy()
        T = test_T_batch[0].cpu().numpy()
        for k in range(args.view_num):
            Viewpoints[k] = (Viewpoints[k] + T[k][None, ...]- center[None, ...]) @ R[k]
        viewpoints_target = Viewpoints.reshape(-1, 3)


        dispoints_cs_list = []
        for i in range(args.iteration):# 所有阶
            viewpoints_pred = np.concatenate(align_viewpoints_list[i], axis=0)

            dispoints_cs_list_onece = []
            for h in args.height:
                index0 = viewpoints_target[:, 2] > h-0.04
                index1 = viewpoints_target[:, 2] < h+0.03
                viewpoints_target_cs = viewpoints_target[index0 & index1]

                index2 = viewpoints_pred[:, 2] > h-0.04
                index3 = viewpoints_pred[:, 2] < h+0.03
                viewpoints_pred_cs = viewpoints_pred[index2 & index3]

                dispoints_cs_list_onece.append(((((viewpoints_pred_cs[:, None, :].repeat(viewpoints_target_cs.shape[0], 1) - viewpoints_target_cs[None, ...])**2).sum(-1)).min(-1))**0.5)
            dispoints_cs_list.append(dispoints_cs_list_onece)

            print(f'stage {i} testing center chamfer error {((((test_Rotary_Center_batch - modified_rotary_center_list[i]) ** 2).sum(-1)) ** 0.5).item()}')

    return dispoints_cs_list


if __name__=='__main__':
    set_seed(SEED=2023)

    args = parse_args()
    print(f'args:{args}')

    test_datasets_path = os.getcwd().split('/20230414方法blade1')[0] + '/' + args.train_val_root_path + '/' + args.test_datasets_mat_path
    print('The test_datasets_mat_file path is: ', test_datasets_path)

    # batch*view_num*point_num*3；batch*view_num*3*3；batch*view_num*3；batch*3；batch*3*3；batch*3
    test_Viewpoints_tensor, test_R_tensor, test_T_tensor, test_Rotary_Center_tensor, R_init, offset_Q = loaddata(test_datasets_path, args.point_num_val2)
    all_modified_rotary_center_dict = sio.loadmat(f'./results_{args.point_num}_testing_results_sample/result_all_modified_rotary_center.mat')

    test_dataset_num = test_Viewpoints_tensor.shape[0]
    print(f'test_dataset_num: {test_dataset_num}')

    result_path = './results_CS_' + str(args.point_num_val2) + '_' + args.test_out_dir
    if os.path.exists(result_path):
        os.system('rm -rf ' + result_path)
    os.makedirs(result_path)

    update_test_cache(args)
