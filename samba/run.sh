#!/usr/bin/env bash
SAMBA_USER=rtfmri
SAMBA_PASSWORD=password
echo -ne "$SAMBA_PASSWORD\n$SAMBA_PASSWORD\n" | smbpasswd -a -s $SAMBA_USER
service smbd start

chown -R rtfmri:rtfmri /mnt/scanner
su - rtfmri -c "python /detect_dicoms.py"