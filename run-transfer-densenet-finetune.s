#!/bin/bash
#
#SBATCH --job-name=transfer-densenet201-finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:30:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ywn202@nyu.edu

module load pytorch/python3.6/0.2.0_3

cd /scratch/ywn202/dog-breed/dog-breed-identification

python3 transfer-densenet201-finetune-2.py