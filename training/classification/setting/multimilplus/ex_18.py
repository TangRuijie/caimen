from setting import *
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}
av_gpus = '0,5,6,7,8,9'
memory = 10
gpu_num = 1
start_time = 60

params_list = []
value_matrix = []
vis_num = -1

add_params('lr', [0.01, 0.001])

add_params('neg_ratio', [3,5,7], [('lr', [0.001])])

add_params('filter_pred_list', ['[0.3, 0.7]', '[0.5,0.7]'], [('neg_ratio', [5])])

params_list.append('resize_dcm_len')
value_matrix.append([36])

params_list.append('coarse_type target_focus')
value_matrix.append([('ex', 0)])

params_list.append('add_coarse_target')
value_matrix.append([1])

params_list.append('add_normal_coarse')
value_matrix.append([1])

params_list.append('loss_type')
value_matrix.append(['balanced'])

params_list.append('pool_mode depth')
value_matrix.append([
    ('gated_attention',18),
                    ])

vis_num = no_vis_show(params_list)

params_list.append('dataset_mode')
value_matrix.append(['ctslicecoarse1008'])

params_list.append('single_slice v_batch_size batch_size')
value_matrix.append([
    (1, 2, 2)
])

params_list.append('num_threads')
value_matrix.append([5])

params_list.append('method_name model')
value_matrix.append([('multimil', 'multimilplus')])

params_list.append('grad_iter_size')
value_matrix.append([10])

params_list.append('consider_time')
value_matrix.append([1])

params_list.append('valid_with_single_slice')
value_matrix.append([0])

params_list.append('v_dataset_type')
value_matrix.append(['ex'])

params_list.append('fine_with_coarse')
value_matrix.append([1])

params_list.append('niter')
value_matrix.append([16])

params_list.append('valid_freq_ratio')
value_matrix.append([0.2])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num
params['stage' + str(stage_ind)]['memory'] = memory
params['stage' + str(stage_ind)]['av_gpus'] = av_gpus
params['stage' + str(stage_ind)]['gpu_num'] = gpu_num
params['stage' + str(stage_ind)]['start_time'] = start_time
