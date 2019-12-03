import asyncio
from functools import partial

import pathmagic
import os
import argparse
import time
import multiprocessing
import subprocess
import shlex
from typing import Sequence, Any
from multiprocessing.dummy import Pool as ThreadPool
from asyncio import ensure_future


from redhawkmaster.rh_big_guns import rh_tiling_gps_equal_filesize

assert pathmagic

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--code', help='Location where is your redhawk code.', required=True)
parser.add_argument('-b', '--bigfile', help='Location where the big file is it.', required=True)
parser.add_argument('-d', '--data', help='Location where is the data you need to run.', required=True)
parser.add_argument('-r', '--results', help='Location where the result data to be.', required=True)
parser.add_argument('-f', '--flow', help='Location where the flow of the jobs is.', required=True)
parser.add_argument('-t', '--template', help='Which template jobs to use.', required=True)
parser.add_argument('-n', '--number', help='Number of tiles to tile the big file.', required=True)
parser.add_argument('-cl', '--core_limit', help='Number of cores to run the tiles through.', required=True)
args = parser.parse_args()

MAX_RUNNERS = int(args.core_limit)

semaphore = asyncio.Semaphore(MAX_RUNNERS)

COMMANDS = []


def run_command(cmd: str) -> str:
    """
    Run prepared behave command in shell and return its output.
    :param cmd: Well-formed behave command to run.
    :return: Command output as string.
    """

    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True,
            cwd=os.getcwd(),
        )

    except subprocess.CalledProcessError as e:
        output = e.output

    return output


@asyncio.coroutine
def run_command_on_loop(loop: asyncio.AbstractEventLoop, command: str) -> bool:
    """
    Run test for one particular feature, check its result and return report.
    :param loop: Loop to use.
    :param command: Command to run.
    :return: Result of the command.
    """
    with (yield from semaphore):
        runner = partial(run_command, command)
        output = yield from loop.run_in_executor(None, runner)
        return output


@asyncio.coroutine
def run_all_commands(command_list: Sequence[str] = COMMANDS) -> None:
    """
    Run all commands in a list
    :param command_list: List of commands to run.
    """
    loop = asyncio.get_event_loop()
    fs = [run_command_on_loop(loop, command) for command in command_list]
    for f in asyncio.as_completed(fs):
        result = yield from f
        ensure_future(process_result(result))


@asyncio.coroutine
def process_result(result: Any):
    """
    Do something useful with result of the commands
    """
    print(result)


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
            out_files += '/results/' + fil + '/' + fil + '_' + l.strip() + '.las '
        i += 1

    if count_out == 1:
        out_files += '/results/' + fil + '/' + fil + '_' + line[-1].strip() + '.las '

    return out_files


def get_in_files(line, count, fil):
    count_in = 0

    in_files = '-i '

    if count == 0:
        in_files += '/data/' + fil + '.las '
        return in_files

    for l in line:

        if line[-1].strip() in l.strip():
            count_in += 1

    if count_in == 1:
        for i in range(0, len(line) - 1):
            in_files += '/results/' + fil + '/' + fil + '_' + line[i].strip() + '.las '
    else:
        in_files += '/results/' + fil + '/' + fil + '_' + line[0].strip() + '.las '

    return in_files


def run_process(cmd):
    # f = open(args.results + "/logs/log" + cmd[0].split(' ')[12].split('/')[2].split('_')[2].split('Tile')[1] + '.out',
    #         "a")
    for cm in cmd:
        # f.write("=====   Job no." + cm.split(' ')[10].split('/')[-1].split('.')[0] + " for tile " +
        #        cm.split(' ')[-4].split('/')[2] + " start   =====\n")
        # print(cm)
        subprocess.call(cm, shell=True)
        # os.system(cm)

        # f.write("=====   Job no." + cm.split(' ')[10].split('/')[-1].split('.')[0] + " for tile " +
        #        cm.split(' ')[-4].split('/')[2] + "      =====\n")
    # f.close()


def parallel(args):
    results = []

    if not os.path.exists(args.results):
        os.mkdir(args.results)

    if not os.path.exists(args.results + '/logs'):
        os.mkdir(args.results + '/logs')
    else:
        os.system('rm ' + args.results + '/logs/*')

    process_count = 0

    for fil in os.listdir(args.data):

        fil = fil.split('.')[0]
        if not os.path.exists(args.results + '/' + fil):
            os.mkdir(args.results + '/' + fil)

    with open(args.flow) as f:
        data = f.readlines()
        count = 0
        for line in data:
            line = line.split(',')
            result = []

            for fil in os.listdir(args.data):

                fil = fil.split('.')[0]

                process_count += 1

                out_files = get_out_files(line, fil)

                in_files = get_in_files(line, count, fil)


                # command = 'echo "=====   Job no.' + line[-1].strip() + ' for tile ' + fil + ' start   ===== $(date)"  >>' \
                #           + args.results + '/logs/log' + str(process_count - 1).zfill(3) + '.out  &&' \
                #           ' docker run -it --rm -v ' + args.code + ':/code -v ' + args.data + ':/data ' \
                #           '-v ' + args.results + ':/results redhawk python3 /code/template_jobs/' + \
                #           args.template + '/' + line[-1].strip() + '.py ' + in_files + out_files + \
                #           ' >> ' + args.results + '/logs/log' + str(process_count - 1).zfill(3) + '.out ' \
                #           ' && echo "=====   Job no.' + \
                #           line[-1].strip() + ' for tile ' + fil + ' end     ===== $(date)" >> ' \
                #           + args.results + '/logs/log' + str(process_count - 1).zfill(3) + '.out '

                command = 'echo "=====   Job no.{} for tile {} start   ===== $(date)"  >> {}/logs/log_{}.out && ' \
                          'docker run -it --rm -v {}:/code -v {}:/data -v {}:/results ' \
                          'redhawk python3 /code/template_jobs/{}/{}.py {} {} >> {}/logs/log_{}.out && ' \
                          'echo "=====   Job no.{} for tile {} end     ===== $(date)" >> ' \
                          '{}/logs/log_{}.out '

                command = command.format(line[-1].strip(), fil, args.results, fil, args.code, args.data, args.results,
                                         args.template, line[-1].strip(), in_files, out_files, args.results, fil,
                                         line[-1].strip(), fil, args.results, fil)

                result.append(command)
            count += 1
            COMMANDS.append(result)

    return COMMANDS

# python3 execute.py -c /home/mcus/workspace/redhawk-pure-python/src
# -b None
# -d /home/mcus/Downloads/ENEL_data
# -r /home/mcus/workspace/RESULTS
# -f /home/mcus/workspace/redhawk-pure-python/src/manager/ENEL_flow
# -t ENEL
# -n 10

    # pool = ThreadPool(int(args.core_limit))
    # pool.map(run_process, [results[0]])
    # pool.close()


if __name__ == '__main__':
    start = time.time()

    if args.bigfile != 'None':
        print("===== Tiling Start =====")
        rh_tiling_gps_equal_filesize(args.bigfile, args.data + '/', no_tiles=args.number)
        print("===== Tiling End   =====")
    print("===== Processing Start =====")
    # print(call_proc("ls ls"))
    # print(parallel(args))
    loop = asyncio.get_event_loop()
    #for res in parallel(args)[3]:
    loop.run_until_complete(run_all_commands(parallel(args)[3]))
    print("===== Processing End   =====")
    end = time.time()
    print("Time: " + str(end - start))
