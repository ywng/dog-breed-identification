#!/bin/bash
#
#SBATCH --job-name=transfer-vgg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:30:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:4

module load pytorch/python3.6/0.2.0_3

cd /scratch/ywn202/dog-breed/dog-breed-identification

python3 transfer-vgg16.py