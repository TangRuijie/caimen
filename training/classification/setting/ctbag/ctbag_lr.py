from setting import no_vis_show
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('neg_ratio lr')
value_matrix.append([
    (10, 1e-6),
    (5, 1e-5),
    (10, 1e-6),
    (5, 1e-5),
    (10, 1e-7),
    (5, 1e-7)
])

params_list.append('target_focus')
value_matrix.append([2])

params_list.append('method_name')
value_matrix.append(['res'])

params_list.append('depth')
value_matrix.append([18])

params_list.append('reinit_data')
value_matrix.append([0])

params_list.append('v_dataset_type')
value_matrix.append(['cq1'])

vis_num = no_vis_show(params_list)

params_list.append('with_warm_up')
value_matrix.append([0])

# params_list.append('load_dir_ind')
# value_matrix.append([9])

params_list.append('batch_size')
value_matrix.append([30])

params_list.append('dataset_mode')
value_matrix.append(['ctbag'])

params_list.append('model')
value_matrix.append(['ctbag'])

params_list.append('load_strict')
value_matrix.append([0])

params_list.append('target_train_num')
value_matrix.append([200])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num


