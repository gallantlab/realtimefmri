FROM ubuntu:18.04
RUN apt-get update \
    && apt-get install -y python3-pip git tmux
    # && rm -rf /var/lib/apt/lists/*

# AFNI
RUN apt-get install -y tcsh xfonts-base python-qt4 \
                       gsl-bin netpbm gnome-tweak-tool \
                       libjpeg62 xvfb xterm vim curl \
                       gedit evince \
                       libglu1-mesa-dev libglw1-mesa \
                       libxm4 build-essential \
                       libcurl4-openssl-dev libxml2-dev \
                       libssl-dev libgfortran3 \
                       gnome-terminal nautilus \
                       gnome-icon-theme-symbolic

RUN ln -s /usr/lib/x86_64-linux-gnu/libgsl.so.23 /usr/lib/x86_64-linux-gnu/libgsl.so.19

RUN mkdir /data
WORKDIR /data
RUN curl -O https://afni.nimh.nih.gov/pub/dist/bin/linux_ubuntu_16_64/@update.afni.binaries
RUN tcsh @update.afni.binaries -package linux_ubuntu_16_64  -do_extras

# pycortex
RUN pip3 install numpy scipy Cython
RUN pip3 install pycortex
RUN python3 -c "import cortex"
COPY pycortex-options.cfg /root/.config/pycortex/options.cfg
COPY pycortex_store /data/pycortex_store

# realtimefmri
RUN pip3 install git+https://github.com/gallantlab/realtimefmri.git
RUN python3 -c "import realtimefmri"
COPY test_dataset /root/.local/share/realtimefmri/datasets/test_dataset

ENV PATH="/root/abin:${PATH}"

WORKDIR /root