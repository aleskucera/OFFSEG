#!/bin/sh
#SBATCH --time=1
ml PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
ml ml torchvision/0.12.0-foss-2021a-CUDA-11.3.1
ml matplotlib/3.4.2-foss-2021a
ml scikit-learn/0.24.2-foss-2021a

python OFFSEG/src/train.py --batch_size 1

