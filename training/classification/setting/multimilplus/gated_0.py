from setting import *
memory = 10
gpu_num = 1
start_time = 60
av_gpus = '0,1,2,3'

params_list.append('net_name')
value_matrix.append(['resnet18'])

params_list.append('lr')
value_matrix.append([1e-2, 1e-3])

params_list.append('load_size')
value_matrix.append([128, 256])

params_list.append('resize_dcm_len')
value_matrix.append([5, 10, 20, 30])

vis_num = no_vis_show(params_list)

params_list.append('class_num')
value_matrix.append([11])

params_list.append('method_name model')
value_matrix.append([('multimil', 'multimilplus')])

params_list.append('v_batch_size batch_size')
value_matrix.append([(2, 2)])

params_list.append('dataset_mode v_dataset_mode')
value_matrix.append([('zgtumor', 'zgtumor')])

params_list.append('loss_type')
value_matrix.append(['balanced'])

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)

add_params('net_name',
           [
               'resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152',
               'resnext101_32x4d', 'resnext101_64x4d',
               'se_resnet50', 'se_resnet101', 'se_resnet152', 'senet154', 'se_resnext101_32x4d', 'se_resnext50_32x4d',
               'densenet121', 'densenet169', 'densenet201', 'densenet161',
               'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
               'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
           )

upload_setting(vis_num = vis_num, memory = memory, av_gpus = av_gpus, gpu_num = gpu_num, start_time = start_time)
