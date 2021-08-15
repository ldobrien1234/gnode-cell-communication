[Cell1_Model.md](Cell1_Model.md) contains files that both simulate the change in a cell's gene expression over time and model the cell's dynamics
using a neural network.

[Cell1_TimeSeries_Simulation.jl](Cell1_Model.md/Cell1_TimeSeries_Simulation.jl) uses the system of differential equations from [Maclean and Franke](https://github.com/maclean-lab/Cell-Cell-Communication) to model the change in gene expression of a cell over time and create a dataset. Then in [neural_ode.py](neural_ode.py), I use a *neural ODE* to learn the dynamics underlying the dataset.

