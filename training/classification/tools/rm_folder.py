import os
import argparse

def count_file(path):
    f_count = 0
    for r, ds, fs in os.walk(path):
        f_count += len(fs)
    return f_count

def rm_vis_empty():
    root_path = 'vis/'
    folders = os.listdir(root_path)
    for fold in folders:
        f_count = count_file(root_path + fold)
        if f_count == 0:
            os.system('rm -r ' + root_path + fold)

def rm_no_model():
    def has_needs(path):
        for r, ds, fs in os.walk(path):
            if 'TEST' in path:
                tmp_fs = [f for f in fs if f.endswith('result.json') or f.endswith('buffer.json')]
                if len(tmp_fs) > 0:
                    return True
            else:
                tmp_fs = [f for f in fs if f.startswith('optimal') and f.endswith('.pth')]
                if len(tmp_fs) > 0:
                    return True
        return False

    root_path = 'checkpoints/'
    folders = os.listdir(root_path)
    for fold in folders:
        has_model = has_needs(root_path + fold)
        if not has_model:
            os.system('rm -r ' + root_path + fold)

def rm_runs():
    check_path = 'checkpoints/'
    run_path = 'runs/'
    check_folders = os.listdir(check_path)
    run_foloders = os.listdir(run_path)

    for run_name in run_foloders:
        has_check = False
        for check_name in check_folders:
            if run_name in check_name:
                has_check = True
        if not has_check:
            os.system('rm -r ' + run_path + run_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--low', default=0.65)
    parser.add_argument('--time', default='')
    cfg = parser.parse_args()

    rm_no_model()
    rm_runs()
    rm_vis_empty()

