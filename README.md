PyTorch Implementation of Multi-Fusion-Graph U-Nets
======================================


About
-----

PyTorch implementation of Multi-Fusion-Graph U-Nets



Installation
------------


Type

    ./run_GNN.sh DATA FOLD GPU
to run on dataset using fold number (1-10).

You can run

    ./run_GNN.sh DD 0 0
to run on DD dataset with 10-fold cross
validation on GPU #0.


Code
----

The detail implementation of Graph U-Net is in src/utils/ops.py.


Datasets
--------

Check the "data/README.md" for the format. 


Results
-------
| Models   | DD              | IMDBMULTI       | PROTEINS        |
| -------- | --------------- | --------------- | --------------- |
| PSCN     | 76.3    | 45.2      | 75.9    |
| DIFFPOOL | 80.6%           | -               | 76.3%           |
| SAGPool  | 76.5%           | -               | 71.9%           |
| GIN      | 82.0     | 52.3  | 76.2|
| g-U-Net  | 83.29 | 56.26 | 78.88  |
| Ours  | 84.04  | 56.47 | 78.88 |


Reference
---------

