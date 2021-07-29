# gnode-cell-communication
In this repository, I created a Graph Neural Ordinary Differential Equation (GNODE) to
model cell communication.

The file Topology_Trajectories.jl was modified slightly from Franke and MacLean (https://github.com/maclean-lab/Cell-Cell-Communication).
I used their model to simulate training data for the GNODE. cell_data_2000.txt contains a sample of training data
generated by their model. The first row in the text file contains inputs, while the second contains output. Each input
and output is a 3 x 4 feature matrix, where each row represents a cell type, and the columns represent the gene expression
levels of GATA1, PU.1, an additional regulatory gene, and some parameter A0. We only use the first three columns when training
the GNODE.
