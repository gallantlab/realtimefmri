FROM bids/base_validator

RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL http://neuro.debian.net/lists/trusty.us-ca.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key adv --recv-keys --keyserver hkp://pgp.mit.edu:80 0xA5D32F012649A5A9 && \
    apt-get update && \
    apt-get install -y afni && \
    apt-get install -y python3.6 python3.6-dev python3-pip && \
    apt-get install -y cython3 python3-numpy python3-scipy python3-numexpr \
                       python3-matplotlib python3-shapely libxml2 git tmux && \
    apt-get remove -y curl && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3.6 -m pip install --upgrade pip

# pycortex
RUN pip3 install pycortex
RUN python3.6 -c "import cortex"
COPY pycortex-options.cfg /root/.config/pycortex/options.cfg
COPY pycortex_store /data/pycortex_store

# realtimefmri
RUN pip3 install git+https://github.com/gallantlab/realtimefmri.git
RUN python3.6 -c "import realtimefmri"
COPY test_dataset /root/.local/share/realtimefmri/datasets/test_dataset

ENV PATH="$HOME/abin:$PATH"
