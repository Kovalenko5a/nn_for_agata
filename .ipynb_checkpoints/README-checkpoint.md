# Introduction

Welcome to project!

<!-- ## Table of contents
1. [Task](#task) -->

### Task <a name="task"></a>
Find method to calculate number of interactions at each segment of each detector of [AGATA](https://www.sciencedirect.com/science/article/pii/S0168900211021516?via%3Dihub), to clarify the pulse-shape analysis (PSA). In this project we use the [Machine learning](https://en.wikipedia.org/wiki/Machine_learning).

### GEANT4 simulation of AGATA 
For simulation we used the most recent branch of GEANT4 model - GANIL's branch. As an example you can find macro files which we used to simulate some data [my_macro_file](my_macro.mac).



### Available input data from GEANT4 simulation
1. Main approach.

   We can receive the deposed energy, time and position of $\gamma$-ray interactions with Germanium (not considering the type of interaction). All interactions sorted in batches within the same time range (100 $\mu s$). You can find example of raw dataset in [OnlyGammaEvents.0000]("gnn_with_prob/OnlyGammaEvents.0000"). Form file like above we extract just iformation we need without any changes.

2. Secondary approach.

    Is about use information about all charge-carriers which borned during interactions to build pulse-shapes (ps) and use instead of deposed energy - the whole ps. Data processing in [nn_with_ps](nn_with_ps) directory (more details later).


# Project details

Creation of Neural Network occured like series of experiments with gradual increasing of "complexity". Some of experiment step left out of project and here you can find only the code which I thought were necessary.

I want to notice that my code might to look a little unprofessional: that's not ok in general but ok for me (just sorry for that).

### First model
All codes of this model you can find in [features_lable_data_format](features_lable_data_format) directory.
Analytical model might be presented as:
$[{Edep, Crystal, Segment} \rangle \overset{f}\longrightarrow Num\_of\_interaction$

For instans of input data you can see [file](features_lable_data_format/out.csv) which was recived from [script](features_lable_data_format/out_creator.py) for raw data processing. This [file](features_lable_data_format/out.csv) we read into [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) in [main code](features_lable_data_format/functional_api_model.py).
Want to notice that in [main code](features_lable_data_format/functional_api_model.py) you can find model with separate input (input1, input2) and concatenated output (out), also in previous versions was used the ["one-hot"](https://en.wikipedia.org/wiki/One-hot) encoding, but I dropped it because in the next version we use the matrix of crystal-segment numbers which mightly the same as one-hot but of next level I suppose.
    
About customisation:
1. Can be changed the number of layers, the number of nodes in each layer. Exception - first and last layer: the input and output shape should be correspond to shapes of feature and target data.
2. The data separated in ratio 3:7 (test_data:train_data). Ratio can be changed, just set the value (0-1) of train_test_split variable in code.
3. You can change hyperparameters:
    * learning_rate - learning rate (for gradient descent)
    * epochs - number of epochs
    * batch_size - size of batch of data for each epoch


### Second model

All codes for this model you can find in [matrix_data_format_nn](matrix_data_format_nn).



