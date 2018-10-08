FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y wget gnupg && \
    wget -O- http://neuro.debian.net/lists/bionic.us-ca.full | tee /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key adv --recv-keys --keyserver pool.sks-keyservers.net 0xA5D32F012649A5A9 && \
    apt-get update && \
    apt-get install -y afni && \
    apt-get install -y python3-dev python3-pip && \
    apt-get install -y cython3 python3-numpy python3-scipy python3-numexpr \
                       python3-matplotlib python3-shapely libxml2 git tmux nano && \
    apt-get remove -y wget && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 -m pip install --upgrade pip
RUN pip3 install redis

# pycortex
RUN pip3 install pycortex
RUN python3 -c "import cortex"
COPY pycortex-options.cfg /root/.config/pycortex/options.cfg
COPY pycortex_store /data/pycortex_store

# realtimefmri
RUN echo "hi4"
RUN pip3 install git+https://github.com/gallantlab/realtimefmri.git
RUN python3 -c "import realtimefmri"
COPY test_dataset /root/.local/share/realtimefmri/datasets/test_dataset
COPY pipelines /root/.local/share/realtimefmri/pipelines

ENV PATH="$HOME/abin:$PATH"

ENTRYPOINT ["realtimefmri"]
