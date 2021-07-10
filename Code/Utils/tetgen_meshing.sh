#!/usr/bin/env bash

set -e

cd /home/dimitris/Documents/Thesis/Models\ with\ Electrodes/meshed/

for file in *.poly; do 
    if [ -f "$file" ]; then 
        time ~/repos/tetgen_v1.5.1/build/tetgen -zpq1.8/0O4a30kNEFAV "$file"
    fi 
done
