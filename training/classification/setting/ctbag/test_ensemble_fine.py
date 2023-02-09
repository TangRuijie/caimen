from setting import no_vis_show
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('sample_list')
value_matrix.append(['[0,1,1,1]', '[0,0,1,1]', '[0,0,0,1]'])

params_list.append('l_state')
value_matrix.append(['valid','test'])

params_list.append('model')
value_matrix.append(['ensemble'])

params_list.append('serial_batches')
value_matrix.append([1])

params_list.append('resize_dcm_len')
value_matrix.append([-1])

params_list.append('v_dataset_mode')
value_matrix.append(['ensemble'])

params_list.append('target_focus')
value_matrix.append([0])

params_list.append('valid_with_normal')
value_matrix.append([0])

vis_num = no_vis_show(params_list)

params_list.append('use_softmax')
value_matrix.append([0])

params_list.append('rgb_sort')
value_matrix.append(['BGR'])

params_list.append('preprocess')
value_matrix.append(['resize'])

params_list.append('original_structure')
value_matrix.append([0])

params_list.append('window_type')
value_matrix.append([1])

params_list.append('v_dataset_type')
value_matrix.append(['ex'])
# value_matrix.append(['cq1'])

# params_list.append('load_dir')
# value_matrix.append(['/home2/heyuwei/cttask_his/his/checkpoints/Apr18_00-27stage0,target_focus=0,load_pretrained_net=1,data_norm_type=competition,method_name=seres,lr=0.01,alpha_mult_ratio=2,single_slice=1,window_type=0,niter=25,fine_with_coarse=0,target_train_num,gpu_ids=4_0.9361780997468729'])

# params_list.append('valid_with_normal')
# value_matrix.append([1])
# params_list.append('load_strict')
# value_matrix.append([0])

# params_list.append('reinit_data')
# value_matrix.append([1])

# params_list.append('loss_type')
# value_matrix.append(['balanced'])

params_list.append('num_thread')
value_matrix.append([3])

params_list.append('single_slice')
value_matrix.append([1])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num
