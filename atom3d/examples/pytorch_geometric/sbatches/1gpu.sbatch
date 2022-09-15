#!/bin/bash
#SBATCH --partition rondror
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-socket=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00

source ~/.gert
$@
