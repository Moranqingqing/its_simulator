#!/bin/sh

remote="googlecloud"
#remote="john"

for i in  06 07 08 09 10 11 12 13 14 15 16 17 18
do
  echo "Port forwarding ... 160$i to 60$i"
  ssh -N -f -L localhost:160$i:localhost:60$i $remote
done

sshfs $remote:$HOME /home/ncarrara/$remote