from setting import no_vis_show
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('method_name')
value_matrix.append(['res'])

params_list.append('depth')
value_matrix.append([50])

params_list.append('target_focus')
value_matrix.append([3])

params_list.append('target_train_num')
value_matrix.append([-1])

params_list.append('model')
value_matrix.append(['ctbag'])

params_list.append('neg_ratio window_type')
value_matrix.append([
    (2, 0),
    (3, 0),
    (3, 1),
    (4, 0)
])

vis_num = no_vis_show(params_list)
# params_list.append('valid_with_normal')
# value_matrix.append([1])
# params_list.append('load_strict')
# value_matrix.append([0])

# params_list.append('reinit_data')
# value_matrix.append([1])

params_list.append('lr')
value_matrix.append([1e-4])

params_list.append('window_type')
value_matrix.append([0])

params_list.append('loss_type')
value_matrix.append(['balanced'])


params_list.append('num_thread')
value_matrix.append([8])

params_list.append('single_slice')
value_matrix.append([1])

params_list.append('valid_freq_ratio')
value_matrix.append([0.5])

params_list.append('with_warm_up')
value_matrix.append([1])

params_list.append('dataset_mode')
value_matrix.append(['ctbag'])
# value_matrix.append(['ctsamplebag'])

params_list.append('batch_size')
value_matrix.append([10])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num
