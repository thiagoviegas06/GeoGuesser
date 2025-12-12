#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=3:00:00
#SBATCH --account=pr_130_general
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=dr3432@nyu.edu
#SBATCH --job-name=CV-GeoLocation
#SBATCH --output=slurm-geolocation-%j.out


if [ -n "$1" ]; then 
	# Looking to resume training from checkpoint
	echo "Model selected: $1"
	ARGS="--model_path $1"

else
	#Training from 0
	ARGS=""

fi 

module purge
	
singularity exec --nv \
	    --overlay /scratch/$USER/computerVision/pytorch_env.ext3:ro \
	    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python test.py $ARGS"
