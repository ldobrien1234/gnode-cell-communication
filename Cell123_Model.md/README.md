# Cell123_Model.md

[Cell123_Model.md](Cell123_Model.md) contains both files that simulate the gene expression of three cells to create a dataset and files that use a *neural ODE* to uncover the dynamics of the cells given time-series data.

[Cell123_TimeSeries_Simulation.jl](Cell123_TimeSeries_Simulation.jl) uses the system of differential equations from [Maclean and Franke](https://github.com/maclean-lab/Cell-Cell-Communication) to model the change in gene expression of three cells over time and create a dataset.

<img width="500" alt="diff_eqs" src="https://user-images.githubusercontent.com/86622061/129943051-0db6013b-cb4e-4ddf-b6ce-7a490c92c23f.PNG">

Then in [neural_ode.py](neural_ode.py), we use a *neural ODE* to learn the dynamics underlying the dataset. [data_classes.py](data_classes.py) is a supportive file that helps our neural network organize the data. In training, we saw the following decrease in loss and MAPE. The blue data represents training and the orange data represents testing.

![cell123_loss](https://user-images.githubusercontent.com/86622061/129656746-ddd5f53b-9f67-4af9-998a-9f4784a4b038.png)
![cell123_mape](https://user-images.githubusercontent.com/86622061/129763457-926aa626-e835-4af3-a074-8161ac6d3e2e.png)


The loss values for each epoch are exported to [cell123_loss.txt](cell123_loss.txt), and the trained model is exported to [Cell123_Model.pt](Cell123_Model.pt). Then, we analyze the accuracy of the model in [model_analysis.py](model_analysis.py). The file chooses a random training example and plots the ground truth versus the model's prediction like below.

![cell123_trajectory1](https://user-images.githubusercontent.com/86622061/129763518-0c44267c-3d5a-4a3a-bca8-6910ce22ca32.png)
![cell123_trajectory2](https://user-images.githubusercontent.com/86622061/129763523-76d13174-ac1d-40ee-92f5-a9b8849ab6cb.png)
![cell123_trajectory3](https://user-images.githubusercontent.com/86622061/129763528-f9c1fc21-dfc2-48b9-a692-467048068f2f.png)



Then, the the file attempts to determine the causal dependencies. We sum the absolute values of the Jacobian of the dynamics with respect to the inputs at each evaluation time as proposed by [Aliee et al.](https://arxiv.org/pdf/2106.12430.pdf) in 2021. We get something like the following heat map. This matrix can help us reconstruct the differential equations used to model the system.

![cell123_causes](https://user-images.githubusercontent.com/86622061/129763553-4a7686b1-63ec-4497-b8d5-f4038106d4a1.png)

