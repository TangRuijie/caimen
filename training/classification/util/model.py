import os
import copy
import torch
import numpy as np
import os.path as osp

def change_dir_name(opt, best_value, visualizer = None):
    value = best_value
    c_save_dir = opt.save_dir

    if value <= 1:
        value_str = '{:.4f}'.format(value)
    else:
        value_str = '{:.3f}'.format(value)

    opt.save_dir = opt.o_save_dir + '_' + value_str
    os.system('mv ' + c_save_dir + ' ' + opt.save_dir)
    if visualizer is not None:
        visualizer.log_name = visualizer.log_name.replace(c_save_dir, opt.save_dir)

    c_log_dir = c_save_dir.replace('checkpoints/', 'runs/')
    new_log_dir = opt.save_dir.replace('checkpoints/', 'runs/')
    os.system('mv ' + c_log_dir + ' ' + new_log_dir)

    if visualizer is not None:
        visualizer.writer.log_dir = new_log_dir

def load_networks(opt, epoch = 'optimal', load_dir = None, strict = True):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    for name in opt.net_names:
        if isinstance(name, str):
            load_filename = '%s_net_%s.pth' % (epoch, name)
            if load_dir is None:
                load_dir = opt.save_dir

            load_path = os.path.join(load_dir, load_filename)

            if not osp.exists(load_path):
                print('net ' + name + ' has no state dict')
                continue

            net = getattr(opt, 'net_' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location = 'cpu')
            net.load_state_dict(state_dict, strict = strict)

def save_networks(opt, epoch, visualizer = None):
    """Save all the networks to the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    def save_nets():
        for name in opt.net_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(opt.save_dir, save_filename)
                net = getattr(opt, 'net_' + name)

                if isinstance(net, torch.nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)

    if epoch != 'optimal':
        save_nets()
        return

    tmp_v_value = opt.valid_metric
    if tmp_v_value > opt.best_m_value:
        opt.best_m_value = tmp_v_value

        if visualizer is not None:
            change_dir_name(opt, visualizer)

        pred_fname = osp.join(opt.save_dir, 'pred_result.txt')
        if osp.exists(pred_fname):
            n_pred_fname = pred_fname.replace('pred','optimal_pred')
            os.system('mv ' + pred_fname + ' ' + n_pred_fname)

        opt.wait_epoch = 0
        save_nets()

    else:
        opt.wait_epoch += 1
        if opt.wait_epoch > opt.patient_epoch:
            opt.wait_over = True


def combine_with_id(opt):
    '''

    Args:
        opt:  options or model

    Returns:

    '''

    buffer_names = copy.copy(opt.buffer_names)
    buffer_names.remove('ginput_ids')
    ginput_ids = opt.buffer_ginput_ids
    id_dict = {}

    buf_data = [getattr(opt, 'buffer_' + name) for name in buffer_names]

    for i, tmp_id in enumerate(ginput_ids):
        if tmp_id not in id_dict.keys():
            id_dict[tmp_id] = []
        id_dict[tmp_id].append(i)

    key_list = list(id_dict.keys())

    # import ipdb
    # ipdb.set_trace()
    for i, buf_list in enumerate(buf_data):
        tmp_buf = np.array(buf_list)
        new_buf = []
        for tmp_id in key_list:
            inds = id_dict[tmp_id]
            new_buf.append(np.mean(tmp_buf[inds], axis=0).tolist())

        setattr(opt, 'buffer_' + buffer_names[i], new_buf)

    opt.buffer_ginput_ids = key_list

