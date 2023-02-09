params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('top_net_name')
value_matrix.append(['identity', 'lstm', 'gcn'])

params_list.append('use_bt_feature')
value_matrix.append([1])

params_list.append('with_fc')
value_matrix.append([0])

params_list.append('target_focus')
value_matrix.append([2])

params_list.append('top_k')
value_matrix.append([7])

params_list.append('depth')
value_matrix.append([18])

params_list.append('niter')
value_matrix.append([25])

params_list.append('duel_item')
value_matrix.append([0])

params_list.append('coarse_type')
value_matrix.append(['hunan'])

params_list.append('v_dataset_type')
value_matrix.append(['hunan'])

params_list.append('lr')
value_matrix.append([1e-4])

params_list.append('model')
value_matrix.append(['lstm'])

params_list.append('dataset_mode')
value_matrix.append(['ctslicecoarse'])

params_list.append('valid_freq_ratio')
value_matrix.append([1])

params_list.append('resize_dcm_len')
value_matrix.append([30])

params_list.append('batch_size')
value_matrix.append(['2'])

params_list.append('num_thread')
value_matrix.append([18])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
