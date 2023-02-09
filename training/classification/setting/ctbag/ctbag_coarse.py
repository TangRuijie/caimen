params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('lr grad_iter_size fc_mult weight_decay weight_len')
value_matrix.append([
                     (1e-5, 20, 10,  1e-5, 4),
                     (1e-5, 20, 10,  1e-5, 3),
                     (1e-4, 20, 10,  1e-4, 3),
                     (1e-5, 20, 10,  1e-5, 3),
                     (1e-4, 20, 10,  1e-5, 3)
                     ])

# params_list.append('slice_y_ratio grad_iter_size lr')
# value_matrix.append([(0, 10, 1e-5),
#                      (0, 10, 1e-4),
#                      (0.5, 10, 1e-4)])
# self.opt.weight_len

params_list.append('duel_item')
value_matrix.append([1])

params_list.append('coarse_type')
value_matrix.append(['301_hunan'])

params_list.append('target_focus')
value_matrix.append([0])

params_list.append('depth')
value_matrix.append([18])

params_list.append('duel_loss_type')
value_matrix.append(['rank'])

params_list.append('method_name')
value_matrix.append(['res'])

params_list.append('valid_freq_ratio')
value_matrix.append([0.2])

params_list.append('model')
value_matrix.append(['multiduel'])

params_list.append('single_slice')
value_matrix.append([0])

params_list.append('with_warm_up')
value_matrix.append([1])

params_list.append('dataset_mode')
value_matrix.append(['ctslicecoarse'])

params_list.append('batch_size')
value_matrix.append([1])

params_list.append('num_thread')
value_matrix.append([4])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
