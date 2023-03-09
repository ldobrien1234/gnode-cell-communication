# gnode-cell-communication
In this repository, I am working to use a *Graph Neural Ordinary Differential Equation (GNODE)* to
model cell communication. **This repository was a summer project that ended abruptly, so I apologize for any incompleteness.**

I will use the model proposed by [Franke and MacLean](https://github.com/maclean-lab/Cell-Cell-Communication) to simulate gene expression over time for a group of blood progenitor cells. The latent state of the system is described by a 3 x 3 matrix. Each row represents a cell and the columns represent the gene expression levels of GATA1, PU.1, and X (a regulatory gene).

Then, I will train a GNODE to learn the cell-communication model. Additionally, I want to show that we can recover the causal structure of the model. By computing the jacobian of the dynamics, we can infer which genes affect each other and which cells affect each other.

The folders [Cell1_Model.md](Cell1_Model.md) and [Cell123_Model.md](Cell123_Model.md) contain simpler cases of what's proposed above. In each file, I use a simplified version of the model from Franke and Maclean, which doesn't incorporate cell-communication. And I use a standard neural ODE, which can't incorporate relationships between cells. Cell1_Model.md models the trajectory of one cell, while Cell123_Model.md models the trajectory of three cells. Both models perform well.
