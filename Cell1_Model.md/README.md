# Cell1_Model.md

[Cell1_Model.md](Cell1_Model.md) contains files that both simulate the change in a cell's gene expression over time and model the cell's dynamics
using a neural network.

[Cell1_TimeSeries_Simulation.jl](Cell1_Model.md/Cell1_TimeSeries_Simulation.jl) uses the system of differential equations from [Maclean and Franke](https://github.com/maclean-lab/Cell-Cell-Communication) to model the change in gene expression of a cell over time and create a dataset. Then in [neural_ode.py](neural_ode.py), we use a *neural ODE* to learn the dynamics underlying the dataset. [data_classes.py](data_classes.py) is a supportive file that helps our neural network organize the data. In training, we saw the following decrease in loss and MAPE.

![cell1_loss](https://user-images.githubusercontent.com/86622061/129511878-0276c5fb-3095-482a-b888-ddd31c1a3043.png)
![cell1_mape](https://user-images.githubusercontent.com/86622061/129511882-4c2a0522-fd90-4aee-b45d-9b6e9ae66917.png)


The loss values for each epoch are exported to [cell1_loss.txt](cell1_loss.txt), and the trained model is exported to [Cell1_Model.pt](Cell1_Model.pt). Then, we analyze the accuracy of the model in [model_analysis.py](model_analysis.py). The file chooses a random training example and plots the ground truth versus the model's prediction like below.

![gene_trajectories](https://user-images.githubusercontent.com/86622061/129513010-a9af5620-2c14-433a-9f2c-5d44b5d840ae.png)


Then, the the file attempts to determine the causal dependencies. We sum the absolute values of the Jacobian of the dynamics with respect to the inputs at each evaluation time as proposed by [Aliee et al.](https://arxiv.org/pdf/2106.12430.pdf) in 2021. We get something like the following heat map.

![causal_dependencies](https://user-images.githubusercontent.com/86622061/129513023-b6765301-7733-4f60-ab65-6ea7008935cd.png)






