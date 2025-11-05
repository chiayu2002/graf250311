#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=shared
#PBS -o output_high_quality.txt
#PBS -e error_high_quality.txt
#PBS -N nerf_hq
cd ~/graf250311

source ~/.bashrc
conda activate graftest

module load cuda-12.4

# 使用高質量配置訓練
python train.py --config /Data/home/vicky/graf250311/configs/high_quality.yaml
