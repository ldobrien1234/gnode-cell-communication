# gnode-cell-communication
In this repository, I created a Graph Neural Ordinary Differential Equation (GNODE) to
model cell communication.

Topology_Trajectories.jl was modified slightly from Franke and MacLean (https://github.com/maclean-lab/Cell-Cell-Communication).
I used their model to simulate gene expression over time for a group of cells. 
cell_data_2000.txt contains 2000 training examples from their model. The first row in the text file contains inputs, 
while the second contains outputs. Each input and output is a 3 x 4 feature matrix. Each row represents a cell type, 
and the columns represent the gene expression levels of GATA1, PU.1, X (a regulatory gene), and a parameter A0. I only 
used the first three columns when training the GNODE.

cell_neuralnet.py trains a GNODE to find the final gene expression of the cells given the initial states and a
graph describing which cells communicate. In observational data, we could infer graph structure based on 
ligand-receptor pairs between cell types. After training the GNODE, the file also computes the Jacobian of the gene dynamics
at the final time. This lets us infer which cell types are influencing others. The neural network can easily be modified to 
incorporate time-series data. With more time-points, the network might be better at learning the gene 
dynamics and causal relationships between cell types.
