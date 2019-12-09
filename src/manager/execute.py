import pathmagic
import os
import argparse
import time
import multiprocessing
import subprocess
import shlex

from multiprocessing.dummy import Pool as ThreadPool

from redhawkmaster.rh_big_guns import rh_tiling_gps_equal_filesize

assert pathmagic

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--code', help='Location where is your redhawk code.', required=True)
parser.add_argument('-b', '--bigfile', help='Location where the big file is it.', required=True)
parser.add_argument('-d', '--data', help='Location where is the data you need to run.', required=True)
parser.add_argument('-r', '--results', help='Location where the result data to be.', required=True)
parser.add_argument('-f', '--flow', help='Location where the flow of the jobs is.', required=True)
parser.add_argument('-t', '--template', help='Which template jobs to use.', required=True)
parser.add_argument('-mb', '--mbpt', help='MB per tile.', required=True)
parser.add_argument('-cl', '--core_limit', help='Number of cores to run the tiles through.', required=True)
args = parser.parse_args()


def call_proc(cmd):
    """ This runs in a separate thread. """
    # subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out, err


def get_out_files(line, fil):
    """
    Construct the output files argument for docker.
    The main thing is multiple output files.
    :param line: Line of the flow
    :param fil: Name of the tile
    :return:
    """

    # How much outputfiles we have
    count_out = 0
    # The argument for the output files.
    out_files = '-o '

    # We count the output files in the line based on:
    # if the final job in a line is consist in some other names
    # e.x. 002,003_1,003_2,003 -> 003 is consist in 003_1 and 003_2
    # which means we have two output files with names 003_1 and 003_2
    for l in line:

        if line[-1].strip() in l.strip():
            count_out += 1
    i = 0
    # construct the names and append the file names
    for l in line:
        if i == count_out:
            break

        if count_out > 1 and l.strip() is not line[-1].strip() and l is not line[0]:
            out_files += '/results/' + fil + '/' + fil + '_' + l.strip() + '.las '
        i += 1
    # If we have only one name e.x. 004,005
    if count_out == 1:
        out_files += '/results/' + fil + '/' + fil + '_' + line[-1].strip() + '.las '

    return out_files


def get_in_files(line, count, fil):
    """
    Construct the input files argument for docker.
    The main thing is multiple input files.
    :param line: Line of the flow
    :param count: if it is the first call then we get from the tiles
    :param fil: name of the tile
    :return:
    """
    count_in = 0

    # argument for the docker input files
    in_files = '-i '

    # First call of the script we get from the data volume
    if count == 0:
        in_files += '/data/' + fil + '.las '
        return in_files

    # Check how much files we have
    for l in line:

        if line[-1].strip() in l.strip():
            count_in += 1

    # The principe goes like this:
    # if we have multiple jobs and only one output
    # that means we have multiple input jobs and one output
    # e.x. 005,003_1,006
    if count_in == 1:
        for i in range(0, len(line) - 1):
            in_files += '/results/' + fil + '/' + fil + '_' + line[i].strip() + '.las '
    else:
        in_files += '/results/' + fil + '/' + fil + '_' + line[0].strip() + '.las '

    return in_files


def run_process(cmd):
    """
    Run the tiles job sequential
    :param cmd:
    :return:
    """
    for cm in cmd:
        subprocess.call(cm, shell=True)


def parallel(args):
    """
    Function that is making the commands that needs to be runned into the docker containers.
    :param args: arguments from the output command.
    :return:
    """

    results = []
    # If the results folder is not there make it.
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    # If the logs folder is not there make if it is delete everything inside
    if not os.path.exists(args.results + '/logs'):
        os.mkdir(args.results + '/logs')
    else:
        os.system('rm '+args.results + '/logs/*')

    # Count how much processes we have
    process_count = 0

    # Read the flow
    with open(args.flow) as f:
        data = f.readlines()
        # Make the folders for each tile and get the locations of the file
        for fil in os.listdir(args.data):
            result = []
            fil = fil.split('.')[0]
            if not os.path.exists(args.results + '/' + fil):
                os.mkdir(args.results + '/' + fil)
            count = 0
            process_count += 1

            # Go through the flow
            for line in data:
                line = line.split(',')

                # Get the output names
                out_files = get_out_files(line, fil)

                # Get the input names
                in_files = get_in_files(line, count, fil)
                count += 1

                # Construct the command
                command = 'echo "=====   Job no.'+line[-1].strip()+' for tile '+fil+' start   ===== $(date)"  >>'\
                          + args.results + '/logs/log' + str(process_count-1).zfill(3) + '.out  &&' \
                          ' docker run -it --rm -v ' + args.code + ':/code -v ' + args.data + ':/data ' \
                          '-v ' + args.results + ':/results redhawk python3 /code/template_jobs/' + \
                          args.template + '/' + line[-1].strip() + '.py ' + in_files + out_files + \
                          ' >> ' + args.results + '/logs/log' + str(process_count-1).zfill(3) + '.out '\
                          ' && echo "=====   Job no.'+line[-1].strip()+' for tile '+fil+' end     ===== $(date)" >> '\
                          + args.results + '/logs/log' + str(process_count-1).zfill(3) + '.out '

                result.append(command)

            results.append(result)

    # Open multiprocessing pool and run the jobs
    pool = multiprocessing.Pool(processes=int(args.core_limit))
    pool.map(run_process, results)
    pool.close()


if __name__ == '__main__':
    start = time.time()

    if args.bigfile != 'None':
        print("===== Tiling Start =====")
        rh_tiling_gps_equal_filesize(args.bigfile, args.data + '/', filesize=int(args.mbpt))
        print("===== Tiling End   =====")

    print("===== Processing Start =====")
    # parallel(args)
    print("===== Processing End   =====")

    end = time.time()
    print("Time: " + str(end - start))
