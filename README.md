GERT
==============================

Geometric Embedding Representations from Transformers.  Implementation of GERT as described in 
[Euclidean Transformers for Macromolecular Structures: Lessons Learned](https://icml-compbio.github.io/2022/papers/WCBICML2022_paper_63.pdf) by DD Liu, L Melo, A Costa, M Vögele, RJL Townshend, and RO Dror.

# General Installation

Navigate to `sbatches/install.sbatch` and point the variable `CONDA_BIN` to your conda binary. You can run `which conda` to identify the location of your conda binary.

Then, navigate to the `gert/src/atom3d_combined` directory and run the following command to install:

```
sbatch ./sbatches/install.sbatch
```

This will allocate a GPU node and submit a batch script to SLURM. The batch script will handle:
- loading the appropriate CUDA/GCC modules,
- creating the conda environment called "gert" (if you want to name your conda environment to something else, change name declared in the environment.yml),
- installing the required libraries.

When the installation is done, activate the conda environment with `conda activate <env_name>`.  You may also need to re-load the correct gcc and cuda versions.

We then need to separately install pytorch-geometric (described at https://github.com/pyg-team/pytorch_geometric).
Use the commands
```
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/${TORCH_VERSION}+${CUDA}.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/${TORCH_VERSION}+${CUDA}.html
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/${TORCH_VERSION}+${CUDA}.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/${TORCH_VERSION}+${CUDA}.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/${TORCH_VERSION}+${CUDA}.html
```
For example, if you have torch 1.7.1 and cuda 11.0 (these version should work after the install script), use
```
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
```
etc.

Now, test that your installation works by running `python3 run.py`.  There should be no error messages.

# Running on compute cluster

`sbatch sbatches/1gpu.sbatch python command.py arg1 arg2 ...`

Make a file like this:
```
vim ~/.gert
```
Populate that file with the following:
```
source <path to conda.sh>
conda deactivate
conda deactivate
module load cuda/10.1.168
conda activate <env_name>
```
