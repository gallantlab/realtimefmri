FROM ubuntu:18.04
RUN apt-get update \
    && apt-get install -y python3-pip git
    # && rm -rf /var/lib/apt/lists/*


RUN pip3 install git+https://github.com/gallantlab/realtimefmri.git

# AFNI
RUN apt-get install -y tcsh xfonts-base python-qt4       \
                       gsl-bin netpbm gnome-tweak-tool   \
                       libjpeg62 xvfb xterm vim curl     \
                       gedit evince                      \
                       libglu1-mesa-dev libglw1-mesa     \
                       libxm4 build-essential            \
                       libcurl4-openssl-dev libxml2-dev  \
                       libssl-dev libgfortran3

RUN apt-get install -y gnome-terminal nautilus          \
                       gnome-icon-theme-symbolic

RUN ln -s /usr/lib/x86_64-linux-gnu/libgsl.so.23 /usr/lib/x86_64-linux-gnu/libgsl.so.19

RUN mkdir /data
WORKDIR /data
RUN curl -O https://afni.nimh.nih.gov/pub/dist/bin/linux_ubuntu_16_64/@update.afni.binaries
RUN tcsh @update.afni.binaries -package linux_ubuntu_16_64  -do_extras

# pycortex
RUN pip3 install pycortex
RUN python3 -c "import cortex"
COPY pycortex-options.cfg /root/.config/pycortex/options.cfg
RUN mkdir -p /data/pycortex_store/RGfs
WORKDIR /data/pycortex_store/
COPY pycortex_store .

RUN pip3 install git+https://github.com/gallantlab/realtimefmri.git -U
RUN apt-get install -y tmux

RUN python3 -c "import realtimefmri"
COPY test_dataset /root/.local/share/realtimefmri/datasets/test_dataset

# device
RUN mkdir /dev/input


WORKDIR ~