from setting import no_vis_show
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('load_dir_ind')
value_matrix.append([13,14])

params_list.append('target_focus')
value_matrix.append([1])

params_list.append('method_name')
value_matrix.append(['res'])

params_list.append('v_dataset_type')
value_matrix.append(['demo'])

# params_list.append('vis_method')
# value_matrix.append(['gradcam'])

# params_list.append('only_target')
# value_matrix.append([1])

vis_num = no_vis_show(params_list)

params_list.append('depth')
value_matrix.append([18])

params_list.append('load_dir_ind')
value_matrix.append([12])
#
params_list.append('model')
value_matrix.append(['ctbag'])

params_list.append('l_state')
value_matrix.append(['valid'])

# params_list.append('load_strict')
# value_matrix.append([1])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num


