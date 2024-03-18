import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim

from utils_train_val.config import parse_args
from utils_train_val.common import loaddata, Alignment, render_points_with_rgb, chamfer_dist, stageprint, adjust_learning_rate, set_seed
from utils_train_val.network import net
from utils_train_val.loss import Loss



def run_train_val(model, optimizer, loss_obj, args):

    used_samples_num, start_pos = args.last_sample_id, args.last_sample_id

    update_val_cache(used_samples_num, model, loss_obj, args)

    while used_samples_num<args.train_max_samples:
        end_pos = start_pos+args.batchsize
        print('Training with pair samples: ' + str(start_pos) + '~' + str(end_pos))
        model = train_one_batch(model, optimizer, loss_obj, start_pos, end_pos, args) # train one batch
        used_samples_num += end_pos-start_pos
        if used_samples_num%(args.val_blank)==0:
            print('validation here, at ' + str(used_samples_num))
            update_val_cache(used_samples_num, model, loss_obj, args) # val once
            torch.save(model.state_dict(), result_path+'/sample_'+str(used_samples_num)+'.pt')

        start_pos = end_pos
        print(f'used_samples_num: {used_samples_num}, train_max_samples: {args.train_max_samples}')

def update_val_cache(used_samples_num, model, loss_obj, args):
    print('updating cache for used_samples_num = ' + str(used_samples_num))
    val_cache_file = result_path + '/result_cache.txt'
    loss_sum_, loss_stages_ = compute_val_loss_values(model, loss_obj, args)
    print('the validation loss: ', loss_sum_, loss_stages_)
    cf = open(val_cache_file, 'a+')
    # The first number is the iteration times.
    cf.write('Steps' + str(used_samples_num // args.batchsize) + ' ')
    cf.write(str(loss_sum_) + ' ')
    for i in range(len(loss_stages_)):
        cf.write(str(loss_stages_[i]) + ' ')
    cf.write('\n')
    cf.close()

    update_pics()

    if args.visualization_while_validation:
        update_visualization(model, args)

def compute_val_loss_values(model, loss_obj, args):
    loss_sum = 0.0
    loss_stages = []
    batch_cnt = 0.0
    print('Computing the validation loss on the validation set:')
    for s in range(0, 2*val_dataset_num, 1): # 测试10个批次
        start_pos = s*args.batchsize
        end_pos = start_pos+args.batchsize

        if end_pos > val_dataset_num:
            end_pos = val_dataset_num

        this_batch_size = end_pos-start_pos
        lsum, lstages = val_one_batch(model, loss_obj, start_pos, end_pos, args)

        if start_pos == 0:
            loss_sum = lsum.item()*this_batch_size
            for i in range(len(lstages)):
                loss_stages.append(lstages[i].item()*this_batch_size)
        else:
            loss_sum += lsum.item() * this_batch_size
            for i in range(len(lstages)):
                loss_stages[i] += lstages[i].item()*this_batch_size

        batch_cnt += this_batch_size

        if end_pos==val_dataset_num:
            break

    print(f'val_dataset_num: {val_dataset_num}; batch_cnt: {batch_cnt}')
    loss_sum /= batch_cnt
    for i in range(len(loss_stages)):
        loss_stages[i] /= batch_cnt

    return loss_sum, loss_stages


def val_one_batch(model, loss_obj, start_pos, end_pos, args):

    val_Viewpoints_batch = val_Viewpoints_tensor[start_pos:end_pos].cuda() # batch*view_num*point_num*3
    val_R_batch = val_R_tensor[start_pos:end_pos].cuda() # batch*view_num*3*3
    val_T_batch = val_T_tensor[start_pos:end_pos].cuda() # batch*view_num*3
    val_Rotary_Center_batch = val_Rotary_Center_tensor[start_pos:end_pos].cuda() # batch*3

    # model.eval()
    with torch.no_grad():
        weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list = model(val_Viewpoints_batch, val_R_batch, val_T_batch, args.view_num)
        regi_loss, regi_loss_stages = loss_obj(weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list, val_Rotary_Center_batch)

    return regi_loss, regi_loss_stages

def update_pics():
    val_cache_file = result_path + '/result_cache.txt'
    cf = open(val_cache_file, 'r')
    lines = cf.readlines()
    x = []
    y_sum = []
    y_cd_on_points = []
    y_cd_on_rotary_center = []

    for i in range(len(lines)):
        index = int(lines[i].split(' ')[0][5:])
        sum_loss = float(lines[i].split(' ')[1])
        cd_on_points_loss = float(lines[i].split(' ')[2])
        cd_on_rotary_center_loss = float(lines[i].split(' ')[3])

        iter_index = index
        x.append(iter_index)
        y_sum.append(sum_loss)
        y_cd_on_points.append(cd_on_points_loss)
        y_cd_on_rotary_center.append(cd_on_rotary_center_loss)

    fig = plt.figure(0)
    fig.clear()
    plt.title('The sum loss')
    plt.xlabel('step')
    plt.ylabel('sum loss')
    plt.plot(x, y_sum, c='red', ls='-')
    plt.savefig(result_path + '/loss_sum.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on cd (points)')
    plt.xlabel('step')
    plt.ylabel('cd loss on points')
    plt.plot(x, y_cd_on_points, c='yellow', ls='-')
    plt.savefig(result_path + '/loss_cd_on_points.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on cd (rotary_center)')
    plt.xlabel('step')
    plt.ylabel('cd loss on rotary_center')
    plt.plot(x, y_cd_on_rotary_center, c='blue', ls='-')
    plt.savefig(result_path + '/loss_cd_on_rotary_center.png')

def update_visualization(model, args):
    val_cache_file = result_path + '/result_cache.txt'
    cf = open(val_cache_file, 'r')
    lines = cf.readlines()
    last_line = lines[len(lines) - 1]
    iter_num = int(last_line.split(' ')[0][5:])
    visual_folder = result_path + '/visual_' + str(iter_num * args.batchsize)
    if not os.path.exists(visual_folder):
        os.mkdir(visual_folder)

    print('Satrt to visualize the results now.')
    for idd in range(1):
        sample_id = idd
        val_Viewpoints_one_sample = val_Viewpoints_tensor[sample_id:sample_id + 1].cuda()  # 1*view_num*point_num*3
        val_R_tensor_one_sample = val_R_tensor[sample_id:sample_id + 1].cuda()  # 1*view_num*3*3
        val_T_tensor_one_sample = val_T_tensor[sample_id:sample_id + 1].cuda()  # 1*view_num*3
        val_Rotary_Center_one_sample = val_Rotary_Center_tensor[sample_id:sample_id + 1].cuda()  # 1*3

        # model.eval()
        with torch.no_grad():
            weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list = model(val_Viewpoints_one_sample, val_R_tensor_one_sample, val_T_tensor_one_sample, args.view_num)
            iter_time = len(weight_list)

            visual_folder_one_sample = visual_folder + '/' + str(sample_id)
            if not os.path.exists(visual_folder_one_sample):
                os.mkdir(visual_folder_one_sample)

            initial_Viewpoints_name = visual_folder_one_sample + '/initial_Viewpoints.obj'
            val_Viewpoints_one_sample_list = []
            val_R_tensor_one_sample_list = []
            val_T_tensor_one_sample_list = []
            for i in range(args.view_num):
                val_Viewpoints_one_sample_list.append(val_Viewpoints_one_sample[:, i, :, :].transpose(1, 2)) # view_num 1*3*point_num
                val_R_tensor_one_sample_list.append(val_R_tensor_one_sample[:, i, :, :]) # view_num 1*3*3
                val_T_tensor_one_sample_list.append(val_T_tensor_one_sample[:, i, :]) # view_num 1*3

            old_dmt_p = Alignment(val_Rotary_Center_one_sample, val_Viewpoints_one_sample_list, val_R_tensor_one_sample_list, val_T_tensor_one_sample_list, args.view_num) # view_num 1*3*point_num

            rgb = {'r': [1, 1, 1, 1, 0, 0, 0, 0], 'g': [0, 0, 1, 1, 0, 0, 1, 1], 'b': [0, 1, 0, 1, 0, 1, 0, 1]}
            render_points_with_rgb(old_dmt_p, rgb, initial_Viewpoints_name)

            cd_points_list = []
            cd_center_list = []
            for stage in range(iter_time):
                stage_Viewpoints_name = visual_folder_one_sample + '/stage_' + str(stage + 1) + '.obj'
                render_points_with_rgb(align_viewpoints_list[stage], rgb, stage_Viewpoints_name) # view_num batch*3*point_num

                output = 0
                for i in range(1, args.view_num):
                    dist1, dist2 = chamfer_dist(align_viewpoints_list[stage][i-1].transpose(-1, -2), align_viewpoints_list[stage][i].transpose(-1, -2))  # batch*3*point_num--->batch*point_num
                    dist1, _ = (-1 * dist1).topk(k=args.loss_top_k, dim=1)  # batch*topk
                    dist2, _ = (-1 * dist2).topk(k=args.loss_top_k, dim=1)
                    dist1 = -1 * dist1
                    dist2 = -1 * dist2
                    output += torch.mean(dist1 + dist2).item() * 0.5 # average distance of point to point

                cd_points_list.append(output / (args.view_num - 1))
                cd_center_list.append(torch.mean((((val_Rotary_Center_one_sample - modified_rotary_center_list[stage]) ** 2).sum(-1)) ** 0.5).item())

            stageprint(cd_points_list, 'update_visualization points chamfer error')
            stageprint(cd_center_list, 'update_visualization center chamfer error')

def train_one_batch(model, optimizer, loss_obj, start_pos, end_pos, args):
    train_Viewpoints_batch = torch.zeros((args.batchsize, args.view_num, args.point_num, 3)).cuda()
    train_R_batch = torch.zeros((args.batchsize, args.view_num, 3, 3)).cuda()
    train_T_batch = torch.zeros((args.batchsize, args.view_num, 3)).cuda()
    train_Rotary_Center_batch = torch.zeros((args.batchsize, 3)).cuda()

    start_pos = start_pos % train_dataset_num
    end_pos = end_pos % train_dataset_num
    if start_pos < end_pos:
        train_Viewpoints_batch = train_Viewpoints_tensor[start_pos:end_pos].cuda()
        train_R_batch = train_R_tensor[start_pos:end_pos].cuda()
        train_T_batch = train_T_tensor[start_pos:end_pos].cuda()
        train_Rotary_Center_batch = train_Rotary_Center_tensor[start_pos:end_pos].cuda()
    else:
        adjust_learning_rate(optimizer)

        bottom = train_dataset_num-start_pos
        top = end_pos
        train_Viewpoints_batch[:bottom] = train_Viewpoints_tensor[start_pos:].cuda()
        train_Viewpoints_batch[bottom:] = train_Viewpoints_tensor[:top].cuda()
        train_R_batch[:bottom] = train_R_tensor[start_pos:].cuda()
        train_R_batch[bottom:] = train_R_tensor[:top].cuda()
        train_T_batch[:bottom] = train_T_tensor[start_pos:].cuda()
        train_T_batch[bottom:] = train_T_tensor[:top].cuda()
        train_Rotary_Center_batch[:bottom] = train_Rotary_Center_tensor[start_pos:].cuda()
        train_Rotary_Center_batch[bottom:] = train_Rotary_Center_tensor[:top].cuda()

    for train_times in range(1):
        model.train()
        weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list = model(train_Viewpoints_batch, train_R_batch, train_T_batch, args.view_num)
        regi_loss, regi_loss_stages = loss_obj(weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list, train_Rotary_Center_batch)

        cd_points_list = []
        cd_center_list = []
        for stage in range(len(weight_list)):
                output = 0
                for i in range(1, args.view_num):
                    dist1, dist2 = chamfer_dist(align_viewpoints_list[stage][i-1].transpose(-1, -2), align_viewpoints_list[stage][i].transpose(-1, -2))  # batch*3*point_num--->batch*point_num
                    dist1, _ = (-1 * dist1).topk(k=args.loss_top_k, dim=1)  # batch*topk
                    dist2, _ = (-1 * dist2).topk(k=args.loss_top_k, dim=1)
                    dist1 = -1 * dist1
                    dist2 = -1 * dist2
                    output += torch.mean(dist1 + dist2).item() * 0.5 # average distance of point to point

                cd_points_list.append(output/(args.view_num - 1))
                cd_center_list.append(torch.mean((((train_Rotary_Center_batch-modified_rotary_center_list[stage])**2).sum(-1))**0.5).item())

        stageprint(cd_points_list, 'training points chamfer error')
        stageprint(cd_center_list, 'training center chamfer error')

        optimizer.zero_grad()
        regi_loss.backward()
        optimizer.step()

        print('optimizer.lr: ', optimizer.state_dict()['param_groups'][0]['lr'])

        return model


if __name__=='__main__':
    set_seed(SEED=2023)

    args = parse_args()
    print(f'args:{args}')

    train_datasets_path = os.getcwd().split('/20230414方法blade1')[0] + '/' + args.train_val_root_path + '/' + args.train_datasets_mat_path
    val_datasets_path = os.getcwd().split('/20230414方法blade1')[0] + '/' + args.train_val_root_path + '/' + args.val_datasets_mat_path
    print('The train_datasets_mat_file path is: ', train_datasets_path)
    print('The val_datasets_mat_file path is: ', val_datasets_path)

    # batch*view_num*point_num*3；batch*view_num*3*3；batch*3
    train_Viewpoints_tensor, train_R_tensor, train_T_tensor, train_Rotary_Center_tensor, _, _ = loaddata(train_datasets_path, args.point_num)
    val_Viewpoints_tensor, val_R_tensor, val_T_tensor, val_Rotary_Center_tensor, _, _ = loaddata(val_datasets_path, args.point_num)

    train_dataset_num = train_Viewpoints_tensor.shape[0]
    val_dataset_num = val_Viewpoints_tensor.shape[0]
    print(f'train_dataset_num: {train_dataset_num}; val_dataset_num: {val_dataset_num}')

    rc_net = net().cuda()

    result_path = './results_' + str(args.point_num) + '_' + args.train_out_dir
    if args.last_sample_id == 0:
        if os.path.exists(result_path):
            os.system('rm -rf ' + result_path)
        os.makedirs(result_path)
    else:
        rc_net.load_state_dict(torch.load(f'{result_path}/sample_{args.last_sample_id}.pt'), True)

    optimizer = optim.AdamW(rc_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.epsilon)

    loss_obj = Loss(args)

    run_train_val(rc_net, optimizer, loss_obj, args)