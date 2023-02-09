from setting import *
memory = 15
gpu_num = 1
start_time = 60
av_gpus = '0,1,2,3'

add_params('dataset_mode v_dataset_mode', [('zgfulltumor', 'zgfulltumor')])

add_params('bbox_ex_ratio1 bbox_ex_ratio2 certain_crop_size load_size channel_comb_type', [
    (1.2, 2, 192, 256, 'add')
    ])

add_params('target_focus', [-1])

add_params('l_state', ['valid'])

add_params('load_net', [1])
vis_num = no_vis_show(params_list)

add_params('num_threads', [6])

add_params('serial_batches', [0])

add_params('load_dir_ind', [0])

add_params('batch_size', [1])

add_params('v_batch_size', [1])

add_params('in_channel', [4])

add_params('embd_pos', [0])

add_params('crop_ratio', [0.3])

add_params('tail_head_num', [2])

#add_params('preprocess', ['resize,flip,rotate'])
add_params('preprocess', ['resize'])

add_params('method_name model', [('fpn', 'transformer')])

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)
