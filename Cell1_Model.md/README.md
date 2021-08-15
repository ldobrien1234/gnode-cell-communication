# Cell1_Model.md

[Cell1_Model.md](Cell1_Model.md) contains files that both simulate the change in a cell's gene expression over time and model the cell's dynamics
using a neural network.

[Cell1_TimeSeries_Simulation.jl](Cell1_Model.md/Cell1_TimeSeries_Simulation.jl) uses the system of differential equations from [Maclean and Franke](https://github.com/maclean-lab/Cell-Cell-Communication) to model the change in gene expression of a cell over time and create a dataset. Then in [neural_ode.py](neural_ode.py), I use a *neural ODE* to learn the dynamics underlying the dataset. [data_classes.py](data_classes.py) is a supportive file that helps our neural network organize the data. In training, we saw the following decrease in loss and MAPE.

![cell1_loss](https://user-images.githubusercontent.com/86622061/129490456-0fa93103-7545-44d7-82c3-c853ef5b6953.png)
![cell1_mape](https://user-images.githubusercontent.com/86622061/129490462-bad052ed-fc58-4a25-bfcc-0f7218808410.png)

The loss values for each epoch are exported to [cell1_loss.txt](cell1_loss.txt), and the trained model is exported to [Cell1_Model.pt](Cell1_Model.pt). Then, I analyze the accuracy of the model in [model_analysis.py](model_analysis.py). The function chooses a random training example and plots the ground truth versus the model's prediction like below.

![cell1_trajectories](https://user-images.githubusercontent.com/86622061/129490603-662b5e74-e086-4170-96fd-50216632421e.png)

Then, the the file attempts to determine the causal dependencies. I sum the absolute values of the Jacobian of the dynamics with respect to the inputs at each evaluation time as proposed by [Aliee et al.](https://arxiv.org/pdf/2106.12430.pdf) in 2021. We get something like the following heat map.

![cell1_causes](https://user-images.githubusercontent.com/86622061/129490688-845d05fd-794d-4472-a31a-d5f40689b803.png)





