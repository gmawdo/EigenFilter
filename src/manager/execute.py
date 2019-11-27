import os
import argparse
import time
import multiprocessing
import subprocess
import shlex

from multiprocessing.pool import ThreadPool

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--code', help='Location where is your redhawk code.', required=True)
parser.add_argument('-d', '--data', help='Location where is the data you need to run.', required=True)
parser.add_argument('-r', '--results', help='Location where the result data to be.', required=True)
parser.add_argument('-f', '--flow', help='Location where the flow of the jobs is.', required=True)
parser.add_argument('-t', '--template', help='Which template jobs to use.', required=True)
args = parser.parse_args()

start = time.time()


def call_proc(cmd):
    """ This runs in a separate thread. """
    # subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out, err


def get_out_files(line, fil):
    count_out = 0
    out_files = '-o '
    for l in line:

        if line[-1].strip() in l.strip():
            count_out += 1
    i = 0
    for l in line:
        if i == count_out:
            break

        if count_out > 1 and l.strip() is not line[-1].strip() and l is not line[0]:
            out_files += '/results/'+fil+'/'+fil+'_' + l.strip() + '.las '
        i += 1

    if count_out == 1:
        out_files += '/results/'+fil+'/'+fil+'_' + line[-1].strip() + '.las '

    return out_files


def get_in_files(line, count, fil):
    count_in = 0

    in_files = '-i '

    if count == 0:
        in_files += '/data/'+fil+'.las '
        return in_files

    for l in line:

        if line[-1].strip() in l.strip():
            count_in += 1

    if count_in == 1:
        for i in range(0, len(line) - 1):
            in_files += '/results/'+fil+'/'+fil+'_' + line[i].strip() + '.las '
    else:
        in_files += '/results/'+fil+'/'+fil+'_' + line[0].strip() + '.las '

    return in_files


def run_process(cmd):

    for cm in cmd:
        os.system(cm)


pool = ThreadPool(multiprocessing.cpu_count())
results = []

if not os.path.exists(args.results):
    os.mkdir(args.results)

process_count = 0
with open(args.flow) as f:
    data = f.readlines()

    for fil in os.listdir(args.data):
        result = []
        fil = fil.split('.')[0]
        if not os.path.exists(args.results+'/'+fil):
            os.mkdir(args.results+'/'+fil)
        count = 0
        process_count += 1
        for line in data:
            line = line.split(',')

            out_files = get_out_files(line, fil)

            in_files = get_in_files(line, count, fil)
            count += 1

            command = 'docker run -v ' + args.code + ':/code ' \
                      '-v ' + args.data + ':/data ' \
                      '-v ' + args.results + ':/results ' \
                      'redhawk python3 /code/template_jobs/' + \
                      args.template + '/' + \
                      line[-1].strip() + '.py ' \
                      + in_files + out_files

            result.append(command)

        results.append(result)

print(results)
print(process_count)

# python3 execute.py -c /home/mcus/workspace/redhawk-pure-python/src -d
# /home/mcus/Downloads/ENEL_data -r /home/mcus/workspace/RESULTS
# -f /home/mcus/workspace/redhawk-pure-python/src/manager/ENEL_flow -t ENEL

pool = multiprocessing.Pool(processes=4)
pool.map(run_process, results)

end = time.time()

print("Time: " + str(end - start))
