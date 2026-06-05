#!/bin/bash

job_name=$1
ml Python/3.11.3-GCCcore-12.3.0

python $HOME/create_job_script.py --server-config-file $HOME/alvis.json --job-config-file  $job_name.json --script-file $job_name.sh
sbatch $job_name.sh