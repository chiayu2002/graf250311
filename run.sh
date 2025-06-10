#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o output05610_SimpleConvTRenderer.txt				
#PBS -e error0610_SimpleConvTRenderer.txt				
#PBS -N nerf
cd ~/graf250520									

source ~/.bashrc											
conda activate graftest	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/graf250520/configs/default.yaml