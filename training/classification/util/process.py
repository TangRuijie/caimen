from subprocess import Popen, PIPE

def ex_pid(p_name):
    process = Popen('ps aux |grep ' + p_name, shell=True, stdout=PIPE)
    stdout, _ = process.communicate()
    stdout = str(stdout)
    p_list = []
    for line in stdout.split('\\n'):
        line = line.rstrip().lstrip()
        if len(line) < 10:
            continue
        if 'grep ' in line:
            continue
        try:
            p = line.split()[1]
        except:
            print('line')
            print(line)

        p_list.append(int(p))
    return p_list