#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** Serial Job in Small Queue***
# 
# Last revised: 22 June 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch clx.serial.slurm" on a Frontera login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J cnn_v4           # Job name
#SBATCH -o cnn_v4.o%j       # Name of stdout output file
#SBATCH -e cnn_v4.e%j       # Name of stderr error file
#SBATCH -p small         # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A DMS22021       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=venu22@tacc.utexas.edu

# Load necessary modules
# module purge
# source $SCRATCH/myenv/bin/activate
# module load python3/3.9.2

# source /scratch1/09052/venurang/myenv/bin/activate

# # Run your Python script
# python3 cnn_v3.py

cd $WORK/climate_modeling
source $SCRATCH/python-envs/ml-env/bin/activate
module load python3/3.9.2
module load ffmpeg
cd /work2/09052/venurang/frontera/climate_modeling/model_T_1/      # Do not use ibrun or any other MPI launcher
python3 -u training.py