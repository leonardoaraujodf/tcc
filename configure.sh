#!/bin/bash

if [ "$(printenv TCC_DIR)" = "" ]
then
   echo "export TCC_DIR=${PWD}" >> ~/.bashrc 
   source ~/.bashrc
fi

if [ -d data/signals ]
then
   echo "Workspace already configured. Nothing to do."
else
   echo "Extracting signal samples..."

   if [ -f data/signals.tar.gz ]
   then
      tar -xvzf data/signals.tar.gz -C data/
   else
      echo "ERROR! File signals.tar.gz don't exist!"
   fi
   echo "Workspace configured!"
fi
