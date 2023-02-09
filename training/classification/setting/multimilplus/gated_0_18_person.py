from setting import *
av_gpus = '3,4,5,6,7'
# av_gpus = '4'
memory = 10
gpu_num = 1
start_time = 300

add_params('target_focus', [0])

add_params('neg_ratio', [3, 5])

add_params('liu_ratio', [4, 1])

add_params('lr', [1e-2, 1e-4])

# add_params('multimil_blood_ratio', [0, 0.5, 1])

add_params('coarse_target_ratio coarse_normal_ratio',
           [(0, 0.3), (0.6, 0.3), (1, 0.6)])

add_params('v_dataset_type', ['person'])

add_params('load_dir_ind', [4])

vis_num = no_vis_show(params_list)

add_params('v_batch_size batch_size', [(2, 2)])

add_params('pool_mode', ['gated_attention'])

add_params('net_name', ['resnet18'])

add_params('load_strict', [0])

# add_params('multi_blood_type', [3])

params_list.append('coarse_type')
value_matrix.append(['301_0,ex'])

params_list.append('avg_resize')
value_matrix.append([0])

params_list.append('rand_aug')
value_matrix.append([1])

params_list.append('resize_dcm_len')
value_matrix.append([30])

params_list.append('filter_pred_list')
value_matrix.append(['[0,0]'])

params_list.append('dataset_mode')
value_matrix.append(['ctslicecoarse1008'])

params_list.append('v_dataset_mode')
value_matrix.append(['ctslice'])

params_list.append('num_threads')
value_matrix.append([6])

params_list.append('method_name model')
value_matrix.append([('multimil_blood', 'multimilplus')])

params_list.append('valid_freq_ratio')
value_matrix.append([0.5])

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)
