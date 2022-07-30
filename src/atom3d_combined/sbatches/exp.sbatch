#!/bin/bash
#SBATCH --partition rondror
#SBATCH --mail-type=FAIL
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=72:00:00
#SBATCH --constraint=GPU_SKU:TITAN_Xp

source ~/.gert
$@
