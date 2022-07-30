#!/bin/bash
#SBATCH --partition rondror
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=5:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --qos=high_p

CONDA_BIN=/path/to/your/conda/binary

module reset
module load cuda/10.1.168
module load gcc/8.1.0

eval "$($CONDA_BIN shell.bash hook)"
conda deactivate
conda deactivate

echo "Creating conda environment..."
conda env create -f environment.yml
echo "Finish creating conda environment..."