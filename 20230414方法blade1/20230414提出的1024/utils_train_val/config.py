import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--point_num', help='number of points',
                        default=2048//2, type=int, metavar='pn')
    parser.add_argument('--point_num_val2', help='number of points',
                        default=2048//1, type=int, metavar='pn')
    parser.add_argument('--iteration', help='iteration times',
                        default=10, type=int, metavar='iter')
    parser.add_argument('--last_sample_id', help='the id in the last saved trained model',
                        default=0, type=int)
    parser.add_argument('--noise_type', choices=['clean', 'jitter'], help='Types of perturbation to consider',
                        default='clean')
    parser.add_argument('--view_num', help='number of viewpoints',
                        default=3, type=int, metavar='vn')
    parser.add_argument('--score_top_k', help='k-nearest neighbor (k-NN)',
                        default=2048//2//20, type=int)
    parser.add_argument('--loss_top_k', help='k-nearest neighbor (k-NN)',
                        default=2048//2//20, type=int)
    parser.add_argument('--val_blank', help='how often the testing process is performed',
                        default=1600, type=int)
    '''train'''
    parser.add_argument('--weight_cd_points',
                        default=0.1, type=float)
    parser.add_argument('--weight_cd_modified_center',
                        default=1, type=float)

    parser.add_argument('--train_val_root_path', help='the path of mat_data (default: None)',
                        default='20230413理论数据集/20230413blade1/Dataset_path_for_proposed_method', metavar='None', type=str)
    parser.add_argument('--train_datasets_mat_path', help='the path of train_datasets_mat_file (default: None)',
                        default='dataset_train_*_2048_*.mat', metavar='None', type=str)
    parser.add_argument('--val_datasets_mat_path', help='the path of val_datasets_mat_file (default: None)',
                        default='dataset_val_*_2048_*.mat', metavar='None', type=str)
    parser.add_argument('--test_datasets_mat_path', help='the path of test_datasets_mat_file (default: None)',
                        default='dataset_test_*_2048_*.mat', metavar='None', type=str)
    parser.add_argument('--train_out_dir', help='the file of the training results',
                        default='training_results_sample', type=str)
    parser.add_argument('--val_out_dir', help='the file of the validation results',
                        default='validation_results_sample', type=str)
    parser.add_argument('--test_out_dir', help='the file of the testing results',
                        default='testing_results_sample', type=str)
    '''train'''
    # '''inference'''
    # parser.add_argument('--weight_cd_points',
    #                     default=1, type=float)
    # parser.add_argument('--weight_cd_modified_center',
    #                     default=-1, type=float)
    # parser.add_argument('--inference_root_path', help='the path of mat_data (default: None)',
    #                     default='20220120三维数据/code-blade1', metavar='None', type=str)
    # parser.add_argument('--inference_datasets_mat_path', help='the path of inference_datasets_mat_file (default: None)',
    #                     default='dataset_inference_*_2048_*.mat', metavar='None', type=str)
    # parser.add_argument('--inference_out_dir', help='the file of the baseline training results',
    #                     default='inference_results_sample', type=str)
    # parser.add_argument('--inference_out_dir_finetune', help='the file of the baseline training results',
    #                     default='inference_results_sample_finetune', type=str)
    # '''inference'''
    # '''noise'''
    # parser.add_argument('--train_val_root_path', help='the path of mat_data (default: None)',
    #                     default='20230413理论数据集/20230413blade1/Dataset_path_for_proposed_method_Noise', metavar='None', type=str)
    # parser.add_argument('--test_datasets_mat_path', help='the path of test_datasets_mat_file (default: None)',
    #                     default='*_0.175.mat', metavar='None', type=str)
    # parser.add_argument('--test_out_dir', help='the file of the testing results',
    #                     default='testing_results_sample_Noise_scale_0.175', type=str)
    # '''noise'''

    parser.add_argument('--learning_rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        default=0.00005, type=float)
    parser.add_argument('--epsilon', type=float,
                        default=1e-8)
    parser.add_argument('--batchsize', help='Batch size for training',
                        default=1, type=int)
    parser.add_argument('--train_max_samples', help='the max number of samples used in the training',
                        default=48000, type=int)
    parser.add_argument('--visualization_while_validation', help='1 if visualize; 0 if not',
                        default=1, type=int, metavar='visual')
    parser.add_argument('--gamma', default=0.8, type=float)

    parser.add_argument('--neighbour_num', help='k-nearest neighbor (k-NN)',
                        default=4, type=int, metavar='nn')
    parser.add_argument('--emb_dims', help='Dimension of embeddings',
                        default=512, type=int, metavar='N')
    parser.add_argument('--n_blocks', help='Num of blocks of encoder&decoder',
                        default=1, metavar='N', type=int)
    parser.add_argument('--ff_dims', help='Num of dimensions of fc in transformer',
                        default=1024, type=int, metavar='N')
    parser.add_argument('--n_heads', help='Num of heads in multiheadedattention',
                        default=1, type=int, metavar='N')
    parser.add_argument('--dropout', help='Dropout ratio in transformer',
                        default=0.0, type=float, metavar='N')

    parser.add_argument('-H', '--height', help='Height of cross-sections (mm)', required=False, default=[42., 53.5, 79.])

    args = parser.parse_args()
    return args