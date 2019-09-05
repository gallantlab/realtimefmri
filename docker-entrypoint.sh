#!/usr/bin/env bash
set -e

# find host ip and write it into hosts file
echo host IP is `/sbin/ip route|awk '/default/ { print $3 }'`
echo writing this into /etc/hosts file

echo -e `/sbin/ip route|awk '/default/ { print $3 }'`'\t' host-machine >> /etc/hosts
echo done. Hosts file looks like this now
cat /etc/hosts


pip3 install -e .
python3 -c "import realtimefmri"

echo "Starting realtimefmri..."
realtimefmri web_interface
