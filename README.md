# Graph-Sampling
Graph Sampling methods using cuAlias.
#### Introduction
1. main-deepwalk.cu is a Graph Sampling algorithm using DeepWalk.
2. main-sage.cu and main-sage-single.cu is a Graph Sampling algorithm using sage.The former uses a warp to process a point, while the latter uses a thread to process a point.
3. main-node2vec-alias-gpu.cu is a Graph Sampling algorithm using Node2vec.

#### Execution of cuAlias
1. download matrix from suitesparse
2. `make`
3. in $run.sh$, change your `PATH` to store the matrix.
4. `sh run.sh`
