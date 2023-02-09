from setting import no_vis_show
params = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

params_list = []
value_matrix = []

params_list.append('depth')
value_matrix.append([50])

params_list.append('method_name')
value_matrix.append(['res'])

params_list.append('l_state')
value_matrix.append(['valid'])

params_list.append('v_dataset_type')
value_matrix.append(['301_3', '301_3v2'])

params_list.append('target_focus')
value_matrix.append([3])

params_list.append('load_dir_ind')
value_matrix.append([1])

vis_num = no_vis_show(params_list)

params_list.append('consider_time')
value_matrix.append([0])

params_list.append('resize_dcm_len')
value_matrix.append([-1])

params_list.append('model')
value_matrix.append(['ctbag'])

# params_list.append('crop_image')
# value_matrix.append([0])

# params_list.append('load_dir')
# value_matrix.append(['/home2/heyuwei/cttask_his/his/checkpoints/Apr18_00-27stage0,target_focus=0,load_pretrained_net=1,data_norm_type=competition,method_name=seres,lr=0.01,alpha_mult_ratio=2,single_slice=1,window_type=0,niter=25,fine_with_coarse=0,target_train_num,gpu_ids=4_0.9361780997468729'])

# params_list.append('valid_with_normal')
# value_matrix.append([0])

params_list.append('use_softmax')
value_matrix.append([1])

# params_list.append('rgb_sort')
# value_matrix.append(['BGR'])

# params_list.append('preprocess')
# value_matrix.append(['resize'])

params_list.append('original_structure')
value_matrix.append([1])

params_list.append('window_type')
value_matrix.append([1])


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

params_list.append('batch_size')
value_matrix.append([10])

params['stage' + str(stage_ind)]['params'] = params_list
params['stage' + str(stage_ind)]['values'] = value_matrix
params['stage' + str(stage_ind)]['vis_num'] = vis_num
