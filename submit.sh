#!/bin/bash
### A name for the job - No spaces allowed
#PBS -N mini_drug_test
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=72:00:00
#PBS -l mem=32gb

# --- Load environment variables from .env file ---

#PBS -o localhost:/home/dpathirana/job.log
#PBS -e localhost:/home/dpathirana/job.err
#PBS -m bae
#PBS -M dpathirana@colgate.edu

cd /Users/dilnipathirana/Downloads/Fall_2025/ML/cosc410-project
. conda activate ml
sleep 1
python model.py