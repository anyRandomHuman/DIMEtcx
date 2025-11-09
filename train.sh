#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --output=/pfs/data6/home/ka/ka_anthropomatik/ka_et4232/workspace/DIMEtcx/outputs/%j.out

module load  devel/cuda/12.8
# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate dime
export MUJOCO_GL=egl

python run_dime.py "$@"

conda deactivate
