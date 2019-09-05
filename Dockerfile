FROM python:3.7-stretch

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget gnupg libxml2 && \
    wget -O- http://neuro.debian.net/lists/stretch.us-ca.full | tee /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key adv --recv-keys --no-tty --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9 && \
    apt-get update && \
    apt-get install -y afni dcm2niix && \
    apt-get install -y inkscape && \
    apt-get remove -y wget && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade pipenv setuptools

WORKDIR /app/realtimefmri

COPY Makefile /app/realtimefmri
COPY Pipfile /app/realtimefmri
COPY Pipfile.lock /app/realtimefmri

ENV PIPENV_DONT_USE_PYENV 1
ENV PIPENV_SYSTEM 1

RUN make requirements

# pycortex
RUN pip3 install pycortex
RUN python3 -c "import cortex"
COPY data/pycortex-options.cfg /root/.config/pycortex/options.cfg

ENV PATH="$PATH:/usr/lib/afni/bin"

EXPOSE 8050

COPY docker-entrypoint.sh /app/realtimefmri

ENTRYPOINT ["/app/realtimefmri/docker-entrypoint.sh"]
