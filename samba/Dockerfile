FROM ubuntu:18.04

RUN apt update && apt install samba -y
RUN mkdir -p /usr/local/samba/var/
ADD smb.conf /etc/samba/smb.conf

RUN adduser --disabled-password --gecos "" rtfmri
RUN mkdir /mnt/scanner
RUN chown rtfmri:rtfmri /mnt/scanner

WORKDIR /
ADD run.sh run.sh
RUN chmod u+x run.sh
CMD ["/run.sh"]
