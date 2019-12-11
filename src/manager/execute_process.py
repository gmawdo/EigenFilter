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
parser.add_argument('-mb', '--mbpt', help='MB per tile.', required=True)
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

    if line[-1].strip() == 'cleanup':
        return out_files + "' '"

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
    if line[-1].strip() == 'cleanup':
        in_files = ''

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


def parallel(args):
    """
    Function that is making the commands that needs to be runned into the docker containers.
    :param args: arguments from the output command.
    :return:
    """
    # If the results folder is not there make it.
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    # If the logs folder is not there make if it is delete everything inside
    if not os.path.exists(args.results + '/logs'):
        os.mkdir(args.results + '/logs')
    else:
        os.system('rm ' + args.results + '/logs/*')

    # Count how much processes we have
    process_count = 0

    # Make the folders for each tile
    for fil in os.listdir(args.data):

        fil = fil.split('.')[0]
        if not os.path.exists(args.results + '/' + fil):
            os.mkdir(args.results + '/' + fil)

    # Read the flow
    with open(args.flow) as f:
        data = f.readlines()
        count = 0
        # for each line in the flow
        for line in data:
            # Separate the jobs
            line = line.split(',')
            result = []

            # List the tile directories
            for fil in os.listdir(args.data):
                # Get the name
                fil = fil.split('.')[0]

                process_count += 1

                # Get the out files
                out_files = get_out_files(line, fil)

                # Get the in files
                in_files = get_in_files(line, count, fil)

                # Make the docker command
                command = 'echo "=====   Job no.{} for tile {} start   ===== $(date)"  >> {}/logs/log_{}.out && ' \
                          'docker run -it --rm -v {}:/code -v {}:/data -v {}:/results ' \
                          'redhawk python3 /code/template_jobs/{}/{}.py {} {} >> {}/logs/log_{}.out && ' \
                          'echo "=====   Job no.{} for tile {} end     ===== $(date)" >> ' \
                          '{}/logs/log_{}.out '

                # Populate it
                command = command.format(line[-1].strip(), fil, args.results, fil, args.code, args.data, args.results,
                                         args.template, line[-1].strip(), in_files, out_files, args.results, fil,
                                         line[-1].strip(), fil, args.results, fil)

                if line[-1].strip() == 'cleanup':
                    command = 'echo "=====   {} for tile {} start   ===== $(date)"  >> {}/logs/log_{}.out && ' \
                              'docker run -it --rm -v {}:/data -v {}:/results ' \
                              'redhawk rm {} >> {}/logs/log_{}.out && ' \
                              'echo "=====   {} for tile {} end     ===== $(date)" >> ' \
                              '{}/logs/log_{}.out '.format(line[-1].strip(), fil, args.results, fil, args.data,
                                                           args.results,
                                                           in_files, args.results, fil, line[-1].strip(), fil,
                                                           args.results, fil)

                    # command.format(line[-1].strip(), fil, args.results, fil, args.data, args.results,
                    #                in_files, args.results, fil, line[-1].strip(), fil, args.results, fil)

                result.append(command)

            count += 1

            COMMANDS.append(result)

    return COMMANDS


if __name__ == '__main__':
    start = time.time()

    if args.bigfile != 'None':
        print("===== Tiling Start =====")
        rh_tiling_gps_equal_filesize(args.bigfile, args.data + '/', filesize=float(args.mbpt))
        print("===== Tiling End   =====")
    print("===== Processing Start =====")

    # print(parallel(args)[3])
    loop = asyncio.get_event_loop()
    for res in parallel(args):
        loop.run_until_complete(run_all_commands(res))

    print("===== Processing End   =====")
    end = time.time()
    print("Time: " + str(end - start))
