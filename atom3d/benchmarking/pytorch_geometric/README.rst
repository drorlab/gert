Using PyTorch Geometric
=======================

PyTorch Geometric is available at https://github.com/rusty1s/pytorch_geometric


Installation
------------

Creating the environment
````````````````````````

Create a separate conda environment::

    conda create --name geometric -c pytorch -c rdkit pip rdkit pytorch=1.5 cudatoolkit=10.2


Installing PyTorch Geometric 
````````````````````````````

You can install it by running::

    conda activate geometric
    pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-geometric

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation (`cu102` if installed as above).

Training example
----------------

The training scripts can be invoked like this::

    python train_qm9.py --target 7 --prefix qm9-u0

where the mapping of target numbers and quantities for QM9 can be looked up in the file qm9_targets.dat.
