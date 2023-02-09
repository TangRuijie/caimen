from . import mkdir
import os.path as osp

def save_manager_info(opt, best_m_value):
    with open(osp.join(opt.save_dir, 'finish'), 'w') as f:
        f.write('1')

    end_dir = osp.join('buffer/training', opt.ppid,  opt.ad_stage)
    mkdir(end_dir)
    end_fname = osp.join(end_dir, 'result_' + opt.name + '_' + str(best_m_value))
    print(end_fname)
    with open(end_fname, 'w') as f:
        f.write(opt.pid + ' ' + opt.save_dir + ' ' + str(best_m_value))