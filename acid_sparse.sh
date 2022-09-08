#!/bin/bash -l
#SBATCH --output=acid_sparse.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --constrain='geforce_rtx_2080_ti|titan_xp'

source /itet-stor/shecai/net_scratch/conda/etc/profile.d/conda.sh
conda activate instant-ngp
python scripts/sparse_from_realestate_format.py --txt_src /media/data6/shengqu/datasets/acid/train/cameras/ --img_src /media/data6/shengqu/datasets/acid/train/frames/ --spa_dst /media/data6/shengqu/datasets/acid_sparse/train