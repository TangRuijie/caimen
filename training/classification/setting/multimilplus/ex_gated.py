from setting import no_vis_show
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []
vis_num = -1

params_list.append('coarse_type target_focus')
value_matrix.append([('ex', 0)])

params_list.append('window_type')
value_matrix.append([0])

params_list.append('depth')
value_matrix.append([101])

params_list.append('preprocess')
value_matrix.append(['noise,clane,contrast'])

vis_num = no_vis_show(params_list)

params_list.append('add_coarse_target')
value_matrix.append([1])

params_list.append('add_normal_coarse')
value_matrix.append([1])

params_list.append('loss_type')
value_matrix.append(['balanced'])

params_list.append('pool_mode')
value_matrix.append([
    ('gated_attention'),
                    ])


params_list.append('single_slice v_batch_size batch_size')
value_matrix.append([
    (1, 1, 1),
])

params_list.append('lr')
value_matrix.append([0.001])

params_list.append('num_thread')
value_matrix.append([10])

params_list.append('method_name model')
value_matrix.append([('multimil', 'multimilplus')])

params_list.append('dataset_mode v_dataset_mode')
value_matrix.append([('ctslicecoarse2508', 'ctslice2508')])

params_list.append('grad_iter_size')
value_matrix.append([10])

params_list.append('consider_time')
value_matrix.append([1])

params_list.append('valid_with_single_slice')
value_matrix.append([0])

params_list.append('resize_dcm_len')
value_matrix.append([30])

params_list.append('v_dataset_type')
value_matrix.append(['ex'])

params_list.append('fine_with_coarse')
value_matrix.append([1])

params_list.append('niter')
value_matrix.append([30])

params_list.append('valid_freq_ratio')
value_matrix.append([0.5])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num

