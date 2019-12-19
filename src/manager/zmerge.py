import os
import subprocess
import time


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


def las_zip(laz_location, las_location, move_location, tile_name, unzip=True):
    """
    Unzip laz file.

    :param move_location: Where to move the laz file if unzip
    :param unzip: if true you unzip, if false you zip
    :param laz_location: Location of the laz file
    :param las_location: Where to output it.
    :param tile_name: Name of the file without extension
    :return:
    """
    command_move = ''
    command = "docker run -v {}:/data -v {}:/data_out pointscene/lastools laszip " \
              "-i /data/{}.las -o /data_out/{}.laz".format(las_location, laz_location, tile_name, tile_name)

    if tile_name == "*":
        command = "ls {}/*.las | xargs -n 1 basename | sed s/\"\.las\"/\"\"/g  | xargs -i " \
                  "docker run -v {}:/data -v {}:/data_out pointscene/lastools" \
                  " laszip -i /data/{}.las -o /data_out/{}.laz". \
            format(las_location, las_location, laz_location, '{}', '{}')

    if unzip:
        command = "docker run -v {}:/data -v {}:/data_out pointscene/lastools laszip " \
                  "-i /data/{}.laz -o /data_out/{}.las".format(laz_location, las_location, tile_name, tile_name)

        if tile_name == "*":
            command = "ls {}/*.laz | xargs -n 1 basename | sed s/\"\.laz\"/\"\"/g  | xargs -i " \
                      "docker run -v {}:/data -v {}:/data_out pointscene/lastools" \
                      " laszip -i /data/{}.laz -o /data_out/{}.las". \
                format(laz_location, laz_location, las_location, '{}', '{}')

        command_move = "mv -f {}/{}.laz {}/".format(laz_location, tile_name, move_location)

        print(run_command(command))
        print(run_command(command_move))
    else:
        print(run_command(command))


def merge_job(las_tile_location, out_location, tile_results, job_number, compressed=True):
    """
    Merge a job  into one tile.
    :param compressed: If true it returns a laz. If false it returns a las.
    :param out_location: Location where the merged file will be.
    :param las_tile_location: location of the las tiles produced.
    :param tile_results: location of the tiles results
    :param job_number: which job to merge
    :return:
    """

    while 1:
        time.sleep(20)

        result = run_command("ls {}/* | grep _{}.las | wc -l".format(tile_results, job_number)).strip()
        no_tiles = run_command("ls {} | wc -l".format(las_tile_location)).strip()

        merge_files = run_command(" ls {}/*/*_{}.las |"
                                  " sed s/'{}'/' -i \/data'/g | tr '\\n' ' '".
                                  format(tile_results, job_number, tile_results.replace("/", "\/"))).strip()
        # print(tile_results)
        # print(result, no_tiles)
        if result == no_tiles:
            out_name = run_command("ls {}/*/*_{}.las | sed s/'_Tile'/'  '/g |"
                                   " sed s/'\/'/' '/g | awk '{}' | tail -1".
                                   format(tile_results, job_number, "{print $(NF-1)}")).strip()
            method = "laz"
            if not compressed:
                method = "las"

            docker_merge = "docker run -v {}:/data -v {}:/data_out pointscene/lastools" \
                           " lasmerge {} -o /data_out/{}_{}.{}". \
                format(tile_results, out_location, merge_files, out_name, job_number, method)
            print(run_command(docker_merge))
            # print("Equal")
            break


if __name__ == '__main__':
    # las_zip(laz_location='/home/mcus/Downloads/ENEL_data',
    #         las_location='/home/mcus/Downloads/ENEL_data',
    #         move_location='/home/mcus/Downloads',
    #         tile_name='BIGFILE_ENEL_Tile000_000',
    #         unzip=True)
    merge_job(las_tile_location='/home/mcus/Downloads/ENEL_data',
              tile_results='/home/mcus/workspace/RESULTS',
              out_location='/home/mcus/workspace',
              job_number='003_1',
              compressed=False)
