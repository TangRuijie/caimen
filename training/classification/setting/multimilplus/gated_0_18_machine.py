from setting import *
av_gpus = '0,5,6,7,8'
memory = 10
gpu_num = 1
start_time = 600

params_list.append('target_focus')
value_matrix.append([0])

params_list.append('pool_mode depth load_dir_ind')
value_matrix.append([
    ('gated_attention',18),
                    ])


add_params('filter_with_machine', [1])

add_params('lr', [0.01])

add_params('neg_ratio', [-1,3])

add_params('coarse_type', ['301_0,machine'])

params_list.append('coarse_target_ratio coarse_normal_ratio')
value_matrix.append([
    (0, 0.3),
    (0.3, 0.3)
])

params_list.append('v_dataset_type')
value_matrix.append(['machine'])

vis_num = no_vis_show(params_list)

add_params('v_batch_size batch_size', [(2, 2)])

add_params('grad_iter_size', [20])

params_list.append('balanced_loss_type')
value_matrix.append(['focal'])

# params_list.append('coarse_type')
# value_matrix.append(['301_1,ex'])

params_list.append('avg_resize')
value_matrix.append([0])

params_list.append('rand_aug')
value_matrix.append([1])

params_list.append('loss_type')
value_matrix.append(['balanced'])

params_list.append('resize_dcm_len')
value_matrix.append([30])

params_list.append('filter_pred_list')
value_matrix.append(['[0,0]'])

params_list.append('dataset_mode')
value_matrix.append(['ctslicecoarse1008'])

params_list.append('v_dataset_mode')
value_matrix.append(['ctslice'])

params_list.append('with_warm_up')
value_matrix.append([1])

params_list.append('num_threads')
value_matrix.append([7])

params_list.append('method_name model')
value_matrix.append([('multimil', 'multimilplus')])

params_list.append('valid_freq_ratio')
value_matrix.append([0.5])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num
params['stage' + str(stage_ind)]['memory'] = memory
params['stage' + str(stage_ind)]['av_gpus'] = av_gpus
params['stage' + str(stage_ind)]['gpu_num'] = gpu_num
params['stage' + str(stage_ind)]['start_time'] = start_time
