# Graph-Sampling
Graph Sampling methods using cuAlias.
#### Introduction
1. main-deepwalk.cu and main-deepwalk-single.cu is a Graph Sampling algorithm using DeepWalk.
2. main-sage.cu and main-sage-single.cu is a Graph Sampling algorithm using sage.The former uses a warp to process a point, while the latter uses a thread to process a point.
3. main-node2vec.cu is a Graph Sampling algorithm using Node2vec.

#### Execution of cuAlias
1. download matrix from suitesparse
2. `make deepwalk` or `make deepwalk-single` or `make sage` or `make sage-single` or `make node2vec`
3. in $run.sh$, change your `PATH` to store the matrix, change your `./deepwalk` or `./sage`
4. `sh run.sh`
5. run `Draw*.py` to Draw pic based on data which form is *.csv 
