# RedHawk Classification

This program is doing automated real time classification of conductors,
 poles and vegetation risk using global cloud compute power.
 
# Instructions

Jobs are create in number from 000 until 200 where each number it has
it own meaning. For now input and output of the files that you want to run
must be changed inside of the python script.

The code is in src folder and it is written in Python 3.7.

Running of just one job is with the following command:

`python3 <number of the job>.py`

#  Job Manager

Under job manager is understood running a sequence of jobs on las files
got from tiling one specific big tile. This manager is under src/manager
folder and it has multiple capabilities. First in order to run this jobs 
it requires to have docker installed on your machine, then building a specific
docker image which will be used for running the jobs in docker and at the end
one python command for managing all of them.

# Install docker

First we need to setup the ubuntu firewall:

`sudo ufw app list`

`sudo ufw allow OpenSSH`

`sudo ufw enable`

`sudo ufw status`

Now lets install docker:

`sudo apt-get remove docker docker-engine docker.io`

`sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common`

`curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -`

`sudo apt-key fingerprint 0EBFCD88`

`sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"`

`sudo apt-get update`

`sudo apt-get install -y docker-ce`

`sudo usermod -aG docker ${USER}`

After these commands check if you have docker installed with:

`docker info`

`docker version`

`docker images`

If everything is working so far lets pull the PDAL container on your box:

`docker pull pdal/pdal`

After this use the Dockerfile in src/manager and build the redhawk container:

`cd src/manager`

`docker build --tag redhawk .`

And check in the docker images do you have redhawk images using:

`docker images`

The output should be similar to this:

|  REPOSITORY | TAG | IMAGE ID|CREATED|SIZE|
|---| --- |--- | --- |---|
|  redhawk | latest  | 1bba7b6529b7  | 20 hours ago | 1.72GB|


# Run job manager

The job manager inside has three things:

1. Dockerfile - used for creating the docker image
2. Flow of the jobs that we are going to use
3. The execute script which is running the flow

The two important scripts are **execute.py** and **execute_process.py***. 
The difference between the two is the **execute.py** is making the jobs command for 
each tile and running the in parallel, whilst the **execute_process.py** is making
one job for all tiles and running them in parallel.

Run the scripts like this:

`python3 execute_process.py -c /home/mcus/workspace/redhawk-pure-python/src -b /home/mcus/Downloads/ENEL_old/BIGFILE_ENEL.las -d /home/mcus/Downloads/ENEL_data -r /home/mcus/workspace/RESULTS -f /home/mcus/workspace/redhawk-pure-python/src/manager/ENEL_flow -t ENEL -mb 15 -cl 20`

Explanation of the arguments:

1. **-c** is the location where your readhawk project is.
2. **-b** is the large file from which to get the tiles. If it is None then no tiling.
3. **-d** is the location where your tiles are.
4. **-r** is the location where the results of the jobs will be put in.
5. **-f** is the location of the flow file.
6. **-t** which template jobs we are going to use.
7. **-n** how much tiles we have if -b is different then None
8. **-cl** is the core limit for **execute.py** and process count for **execute_process.py**

Note: all of these arguments are required.