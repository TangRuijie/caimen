from setting import *
memory = 15
gpu_num = 1
start_time = 60
av_gpus = '0,1,2,3'

add_params('loss_type', ['balanced'])

# add_params('multi_task', [1])
# add_params('channel_comb_type', ['add','no'])

# add_params('multi_task_ratio', [1, 0.5,2])

add_params('consider_distance', [0])


# add_params('bbox_ex_ratio1 bbox_ex_ratio2 certain_crop_size', [(1.2, 4, -1),(1.2, 3, -1), (1.2, 2,-1), (1.2, 2,128), (1.2, 2, 96)])
add_params('bbox_ex_ratio1 bbox_ex_ratio2 certain_crop_size load_size channel_comb_type', [
    (1.2, 2, 192, 256, 'add')
    ])


add_params('repeat_iter', [1])

add_params('target_focus', [-1])

add_params('weight_decay', [1e-5])

add_params('batch_size', [3])

add_params('v_batch_size', [1])

add_params('in_channel', [4])

add_params('embd_pos', [0])

add_params('crop_ratio', [0.3])

add_params('tail_head_num', [2])

vis_num = no_vis_show(params_list)

add_params('num_threads', [2])

add_params('serial_batches', [0])

add_params('load_net', [1])

add_params('load_dir_ind', [0])

add_params('preprocess', ['resize'])
#add_params('preprocess', ['resize'])

params_list.append('method_name model')
value_matrix.append([('fpn', 'transformer')])

params_list.append('dataset_mode v_dataset_mode')
value_matrix.append([('zgfulltumor', 'zgfulltumor')])

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)
# add_params('net_name',
#            [
#                'resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152',
#                'resnext101_32x4d', 'resnext101_64x4d',
#                'se_resnet50', 'se_resnet101', 'se_resnet152', 'senet154', 'se_resnext101_32x4d', 'se_resnext50_32x4d',
#                'densenet121', 'densenet169', 'densenet201', 'densenet161',
#                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
#                'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
#            )

#add_params('net_name', ['efficientnet_b7'])

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)

add_params('lr', [1e-2, 1e-3])

add_params('head_comb_mode', ['cat', 'mean'])

add_params('block_comb_mode', ['cat', 'mean'])

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)

#add_params('loss_type', ['cross', 'balanced'])

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)
