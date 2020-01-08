FROM pdal/pdal@sha256:409aec41a953ea5a1af3643b0d5e60457de3ee0cd392c090992de39438fb90b9
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get \
install -y --fix-missing --no-install-recommends \
python3-pip python3.7-dev python3.7 python-matplotlib \
gnuplot && \
pip3 install --upgrade setuptools && \
pip3 install \
sklearn \
pandas \
laspy \
matplotlib \
numba \
pyshp \
numpy==1.16.2 --upgrade && \
apt-get autoremove -y && \
    apt-get clean


