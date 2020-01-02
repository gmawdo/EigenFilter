import pathmagic
import os
import argparse
import time
import multiprocessing
import subprocess
import shlex

from multiprocessing.dummy import Pool as ThreadPool

from manager.zmerge import las_zip, merge_job
from redhawkmaster.rh_big_guns import rh_tiling_gps_equal_filesize

assert pathmagic

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--code', help='Location where is your redhawk code.', required=True)
# parser.add_argument('-d', '--data', help='Location where is the data you need to run.', required=True)
parser.add_argument('-r', '--results', help='Location where the result data to be.', required=True)
parser.add_argument('-f', '--flow', help='Name of the flow for the jobs.It must be located inside template.', required=True)
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
            if line[i].strip() == '000':
                in_files += '/data/' + fil + '.las '
            else:
                in_files += '/results/' + fil + '/' + fil + '_' + line[i].strip() + '.las '
    else:
        in_files += '/results/' + fil + '/' + fil + '_' + line[0].strip() + '.las '

    return in_files


global sema
sema = multiprocessing.Semaphore(int(args.core_limit))


def run_process(cmd):
    """
    Run the tiles job sequential
    :param cmd:
    :return:
    """
    sema.acquire()
    print(sema)
    for cm in cmd:
        subprocess.call(cm, shell=True)
    sema.release()


def parallel(args, big_file):
    """
    Function that is making the commands that needs to be runned into the docker containers.
    :param args: argumenpool_tilests from the output command.
    :return:
    """

    results = []
    # If the results folder is not there make it.
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    if not os.path.exists(args.results + '/' + big_file):
        os.mkdir(args.results + '/' + big_file)

    # If the logs folder is not there make if it is delete everything inside
    if not os.path.exists(args.results + '/' + big_file + '/logs'):
        os.mkdir(args.results + '/' + big_file + '/logs')
    else:
        os.system('rm ' + args.results + '/' + big_file + '/logs/*')

    # Count how much processes we have
    process_count = 0

    # Read the flow
    with open(args.code + '/template_jobs/' + args.template + '/' + args.flow) as f:
        data = f.readlines()
        # Make the folders for each tile and get the locations of the file
        for fil in os.listdir(args.results + '/200_TILES/' + big_file + '/'):
            result = []
            fil = fil.split('.')[0]
            if not os.path.exists(args.results + '/' + big_file + '/' + fil):
                os.mkdir(args.results + '/' + big_file + '/' + fil)
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
                command = 'echo "=====   Job no.{} for tile {} start   ===== $(date)"  >> {}/logs/log_{}.out && ' \
                          'docker run --rm -v {}:/code -v {}:/data -v {}:/results ' \
                          'redhawk python3 /code/template_jobs/{}/{}.py {} {} >> {}/logs/log_{}.out && ' \
                          'echo "=====   Job no.{} for tile {} end     ===== $(date)" >> ' \
                          '{}/logs/log_{}.out '
                command = command.format(line[-1].strip(), fil, args.results + '/' + big_file, fil, args.code,
                                         args.results + '/200_TILES/' + big_file + '/', args.results + '/' + big_file,
                                         args.template, line[-1].strip(), in_files, out_files,
                                         args.results + '/' + big_file, fil,
                                         line[-1].strip(), fil, args.results + '/' + big_file, fil)

                if line[-1].strip() == 'cleanup':
                    command = 'echo "=====   {} for tile {} start   ===== $(date)"  >> {}/logs/log_{}.out && ' \
                              'docker run --rm -v {}:/data -v {}:/results ' \
                              'redhawk rm {} >> {}/logs/log_{}.out && ' \
                              'echo "=====   {} for tile {} end     ===== $(date)" >> ' \
                              '{}/logs/log_{}.out '.format(line[-1].strip(), fil, args.results + '/' + big_file, fil,
                                                           args.results + '/200_TILES/' + big_file + '/',
                                                           args.results + '/' + big_file,
                                                           in_files, args.results + '/' + big_file, fil,
                                                           line[-1].strip(), fil,
                                                           args.results + '/' + big_file, fil)

                # command = 'echo "=====   Job no.'+line[-1].strip()+' for tile '+fil+' start   ===== $(date)"  >>'\
                # + args.results + '/logs/log' + str(process_count-1).zfill(3) + '.out  &&' \ ' docker run -it --rm
                # -v ' + args.code + ':/code -v ' + args.results + '/200_TILES/' + ':/data ' \ '-v ' + args.results +
                # ':/results redhawk python3 /code/template_jobs/' + \ args.template + '/' + line[-1].strip() + '.py
                # ' + in_files + out_files + \ ' >> ' + args.results + '/logs/log' + str(process_count-1).zfill(3) +
                # '.out '\ ' && echo "=====   Job no.'+line[-1].strip()+' for tile '+fil+' end     ===== $(date)" >>
                # '\ + args.results + '/logs/log' + str(process_count-1).zfill(3) + '.out '

                result.append(command)

            results.append(result)

    # print(results)
    # Open multiprocessing pool and run the jobs
    for res in results:
        multiprocessing.Process(target=run_process, args=(res,)).start()
    # pool_tiles.map(run_process, results)
    # pool_tiles.join()


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def get_last_job():
    fileHandle = open(args.flow, 'r')
    lineList = fileHandle.readlines()
    if lineList[-1].split(',')[-1] == 'cleanup':
        if lineList[-2].strip().split(',')[-1] in lineList[-2].strip().split(',')[-2]:
            return lineList[-2].strip().split(',')[-2]
    if lineList[-1].strip().split(',')[-1] in lineList[-1].strip().split(',')[-2]:
        return lineList[-1].strip().split(',')[-2]
    return lineList[-1].strip().split(',')[-1]


def init_flow(infile):
    global sorted_filename_list
    sorted_filename_list = infile


def run_a_flow(item):
    big_file = sorted_filename_list[item].split('.')[0]
    if not os.path.exists(args.results + '/200_TILES/' + big_file):
        os.mkdir(args.results + '/200_TILES/' + big_file)

    print("===== Unlaz Start =====")
    las_zip(laz_location=args.results + '/000_RECEIVED_DATA_LAZ',
            las_location=args.results + '/100_BIG_FILES_LAS',
            move_location=args.results + '/001_RECEIVED_DATA_LAZ_COMPLETE',
            tile_name=big_file,
            unzip=True)
    print("===== Unlaz End   =====")
    # if args.bigfile != 'None':

    print("===== Tiling Start =====")
    rh_tiling_gps_equal_filesize(args.results + '/100_BIG_FILES_LAS/' + big_file + '.las',
                                 args.results + '/200_TILES/' + big_file + '/', filesize=float(args.mbpt))
    print("===== Tiling End   =====")

    print("===== Processing Start =====")
    parallel(args, big_file)
    print("===== Processing End   =====")

    print("===== Merge Start =====")
    multiprocessing.Process(target=merge_job, args=(args.results + '/200_TILES/' + big_file + '/',
                                                    args.results + '/300_PRODUCTS',
                                                    args.results + '/' + big_file + '/',
                                                    get_last_job(),
                                                    True,)).start()
    # print(get_last_job())
    # merge_job(las_tile_location=args.results + '/200_TILES/' + big_file + '/',
    #           tile_results=args.results + '/' + big_file + '/',
    #           out_location=args.results + '/300_PRODUCTS/',
    #           job_number=get_last_job(),
    #           compressed=True)
    print("===== Merge End   =====")


if __name__ == '__main__':

    if not os.path.exists(args.results):
        os.mkdir(args.results)

    if not os.path.exists(args.results + '/000_RECEIVED_DATA_LAZ'):
        os.mkdir(args.results + '/000_RECEIVED_DATA_LAZ')

    if not os.path.exists(args.results + '/001_RECEIVED_DATA_LAZ_COMPLETE'):
        os.mkdir(args.results + '/001_RECEIVED_DATA_LAZ_COMPLETE')

    if not os.path.exists(args.results + '/100_BIG_FILES_LAS'):
        os.mkdir(args.results + '/100_BIG_FILES_LAS')

    if not os.path.exists(args.results + '/200_TILES'):
        os.mkdir(args.results + '/200_TILES')

    if not os.path.exists(args.results + '/300_PRODUCTS'):
        os.mkdir(args.results + '/300_PRODUCTS')

    while 1:
        time.sleep(5)
        start = time.time()
        name_list = os.listdir(args.results + '/000_RECEIVED_DATA_LAZ/')
        full_list = [os.path.join(args.results + '/000_RECEIVED_DATA_LAZ/', i) for i in name_list]
        time_sorted_list = sorted(full_list, key=os.path.getmtime)
        sorted_filename_list = [os.path.basename(i) for i in time_sorted_list]
        # print(sorted_filename_list[0].split('.')[0])
        # poolTiles = multiprocessing.Pool(processes=int(args.core_limit))
        if sorted_filename_list:
            # pool = multiprocessing.Pool(processes=len(sorted_filename_list), initializer=init_flow,
            #                             initargs=(sorted_filename_list,))
            pool = MyPool(processes=2, initializer=init_flow,
                          initargs=(sorted_filename_list,))

            pool.map(run_a_flow, range(len(sorted_filename_list)))

        end = time.time()
        print("Time: " + str(end - start))

    pool.close()
    pool.join()
