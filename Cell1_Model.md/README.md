[Cell1_Model.md](Cell1_Model.md) contains files that both simulate the change in a cell's gene expression over time and model the cell's dynamics
using a neural network.

[Cell1_TimeSeries_Simulation.jl](Cell1_Model.md/Cell1_TimeSeries_Simulation.jl) uses the system of differential equations from [Maclean and Franke](https://github.com/maclean-lab/Cell-Cell-Communication) to model the change in gene expression of a cell over time and create a dataset. Then in [neural_ode.py](neural_ode.py), I use a *neural ODE* to learn the dynamics underlying the dataset. In training, we saw the following decrease in loss and MAPE.

![cell1_loss](https://user-images.githubusercontent.com/86622061/129490456-0fa93103-7545-44d7-82c3-c853ef5b6953.png)
![cell1_mape](https://user-images.githubusercontent.com/86622061/129490462-bad052ed-fc58-4a25-bfcc-0f7218808410.png)




