from setting import no_vis_show
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('method_name')
value_matrix.append(['rsna'])

params_list.append('rsna_method load_size')
value_matrix.append(
    [
        ('dense169', 256),
         ('seres', 256),
    ('dense121', 512),
      ]
                    )

params_list.append('no_neck')
value_matrix.append([1])

params_list.append('l_state')
value_matrix.append(['valid'])

params_list.append('v_dataset_type')
value_matrix.append(['301_0'])

vis_num = no_vis_show(params_list)

params_list.append('crop_image')
value_matrix.append([0])

params_list.append('resize_dcm_len')
value_matrix.append([-1])

params_list.append('valid_with_normal')
value_matrix.append([1])

params_list.append('rgb_sort')
value_matrix.append(['BGR'])

params_list.append('preprocess')
value_matrix.append(['resize'])

params_list.append('window_type')
value_matrix.append([1])

params_list.append('target_focus')
value_matrix.append([0])

params_list.append('model')
value_matrix.append(['ctbag'])

params_list.append('num_thread')
value_matrix.append([3])

params_list.append('single_slice')
value_matrix.append([1])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num
