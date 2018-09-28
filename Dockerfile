FROM ubuntu:18.04
RUN apt-get update \
    && apt-get install -y python3-pip git
    # && rm -rf /var/lib/apt/lists/*
RUN pip3 install git+https://github.com/gallantlab/realtimefmri.git
