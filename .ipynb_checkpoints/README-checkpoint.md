# Introduction

Welcome to project!

<!-- ## Table of contents
1. [Task](#task) -->

## Task <a name="task"></a>
Find method to calculate number of interactions at each segment of each detector of [AGATA](https://www.sciencedirect.com/science/article/pii/S0168900211021516?via%3Dihub), to clarify the pulse-shape analysis (PSA). In this project we use the [Machine learning](https://en.wikipedia.org/wiki/Machine_learning).

## GEANT4 simulation of AGATA 
For simulation we used the most recent branch of GEANT4 model - GANIL's branch. As an example you can find macro files which we used to simulate some data [my_macro_file](my_macro.mac).



## Available input data from GEANT4 simulation
1. Main approach <a name="main"></a>.

   We can receive the deposed energy, time and position of $\gamma$-ray interactions with Germanium (not considering the type of interaction). All interactions sorted in batches within the same time range (100 $\mu s$). You can find example of raw dataset in [OnlyGammaEvents.0000](gnn_with_prob/OnlyGammaEvents.0000). Form file like above we extract just iformation we need without any changes.

2. Secondary approach.

    Using of information about all charge-carriers which borned during interactions to build pulse-shapes (ps) and use instead of deposed energy - the whole ps. Data processing in [nn_with_ps](nn_with_ps) directory (more details later).


# Project details

Creation of Neural Network occured like series of experiments with gradual increasing of "complexity". Some of experiment step left out of project and here you can find only the code which I thought were necessary.

I want to notice that my code might to look a little unprofessional: that's not ok in general but ok for me (just sorry for that).

## First model
All codes of this model you can find in [features_lable_data_format](features_lable_data_format) directory.
Analytical model might be presented as:  
<!-- $[{Edep, Crystal, Segment} \rangle \overset{f}\longrightarrow Num\_of\_interaction$   -->

<img src="https://render.githubusercontent.com/render/math?math=[{Edep,Crystal,Segment}\rangle\overset{f}%20\longrightarrow%20Num\_of\_interaction">

For instans of input data you can see [file](features_lable_data_format/out.csv) which was recived from [script](features_lable_data_format/out_creator.py) for raw data processing. This [file](features_lable_data_format/out.csv) we read into [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) in [main code](features_lable_data_format/functional_api_model.py).
Want to notice that in [main code](features_lable_data_format/functional_api_model.py) you can find model with separate input (input1, input2) and concatenated output (out), also in previous versions was used the ["one-hot"](https://en.wikipedia.org/wiki/One-hot) encoding, but I dropped it because in the next version we use the matrix of crystal-segment numbers which mightly the same as one-hot but of next level I suppose.
    
About customisation:
1. Can be changed the number of layers, the number of nodes in each layer. Exception - first and last layer: the input and output shape should be correspond to shapes of feature and target data.
2. The data separated in ratio 3:7 (test_data:train_data). Ratio can be changed, just set the value (0-1) of train_test_split variable in code.
3. You can change hyperparameters:
    * learning_rate - learning rate (for gradient descent)
    * epochs - number of epochs
    * batch_size - size of batch of data for each epoch
4. Can be implemented some extra activation functions on each layer except the last one (gives always the "1" as output).

To see how it process just run the [main code](features_lable_data_format/functional_api_model.py) in virtual invironment of [this derictory](features_lable_data_format):
```
pipenv install

pipenv run python functional_api_model.py

or

pipenv shell  
python functional_api_model.py
```
With current parameters the output will be like that:
![](features_lable_data_format/Figure_1.png)

And test-value of accuracy around 67% - it is an evidence that model all the time gives 1 as number of interactions.

---
**NOTE**

[If you use Jupyter you can create kernels for each virtenv and use all of them in Jupyter Lab for example.](https://towardsdatascience.com/virtual-environments-for-data-science-running-python-and-jupyter-with-pipenv-c6cb6c44a405)

---

## Second model

All codes for this model presence at [matrix_data_format_nn](matrix_data_format_nn) directory. This model is also about [main aproach](#main) in data format.

Main idea was to use the matrix \(crystal*segment\) with deposed energy in each cell. Of cause for we need separate matrix for each time-range. As target data we use matrix with the same shape but filled with numbers of interactions in each cell.

Illustration to matrix data format:


![Illustration to matrix data format](frame_matrix.png)
<!-- <img src="frame_matrix.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" /> -->

In directory you can find two last experiments:
1. One [convolution layer](https://keras.io/api/layers/convolution_layers/convolution1d/) (as for photos) and using of [pooling](https://keras.io/api/layers/pooling_layers/max_pooling1d/) and [reashaping](https://keras.io/api/layers/reshaping_layers/permute/) (just like turne the rectangle on 90 degree).
2. The same as previous but with separate processing of different parts of matrix (one more dimension of input matrix with numbers of crystals in each cell (in one-hot encoding)).

Just like for previouse model, you just need to install all the requirenments of virtenv, choose the variant of code ([m3_2dConv.py](matrix_data_format_nn/m3_2dConv.py) or [m4_flatte_reshape_pool.py](matrix_data_format_nn/m4_flatte_reshape_pool.py)) and run it in curent virtenv.

Input data matrices generated when you run the main code so it may take a while.  
**...Loss and accuracy function will show later**


## Third model

So the third model in which I'm puting the best hope. The main characteristic of this model is so called [Graphs Neural Network](https://graphneural.network/). Here we can manually connect different interactions which occurred at the same time range with some extra information like distance between segments where we have deposet energy.  
For example, in case of three nodes (segments with signal), graph will be look like that:
![](gnn_with_prob/gnn.png)

**Just for example you can look at [json file with graphs](gnn_with_prob/data1.json) which was created for demonstration, for processing we use [another](gnn_with_prob/data_global_1.json).**

As well as previouse it's also the latest models and thay also contain branching (experiments):
1. Input is just graphs with deposed energy in nodes, distances as conecntionss. Output - vector of number of interactions. Here we implemented the probability distributions at the dense layers for the first time. The main code you can find at the [first directory](gnn_with_prob/1_add_distribution_in_layer).
2. Same as previous but with discrete distribution of number of interactions instead of exact meaning. Here as loss funciton we use so called [Categorical Crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy). See the [second directory](gnn_with_prob/2_add_crossentropy) with main code.
3. Same as the second but with extra information about connections - not only the distances but also the direction from one segment to the other. The last one - [third directory](gnn_with_prob/3_add_directive_vector_in_graph) where you can find relevant codes.

### Let's begin from main directory of third model

[Here](gnn_with_prob) you can find [GammaEvents.0000](gnn_with_prob/GammaEvents.0000) generated on AGATA simulation. This file contain information about relative position of segments in each detector and about relative position of detectors in the AGATA. In [code](gnn_with_prob/segments_dist.py) was exatracted this information and saved in [csv file](gnn_with_prob/buff2.csv) and also added function to calculate distanses and directions.


Also you can see the [test code](gnn_with_prob/experements_distrib.py) to explore how number of interactions distributed. With usage of ProbabilityTensorFlow module we can build the discrete distribution.


Ready to work data sets can be received from [raw data](gnn_with_prob/OnlyGammaEvents.0000) by applying the "json_creator" codes in secodary directories.

### Content of secondary directories

#### First
The [main code](gnn_with_prob/1_add_distribution_in_layer/gnn_for_agata.py) for calculation can recive the input data with help of json-read-method for data_gloval_1.json or as output of function defined in [code for raw data processing](gnn_with_prob/1_add_distribution_in_layer/for_gnn_with_tfp.py) (second variant may take a while).

Remarkable characteristics of main code:
- Config stage contain definition of hyperparameters which can be customized (see "about customisation" in paragraph "first model").
- In load data stage you can see that My Dataset class used to create the array of all available graphs readed from data_global_1.json. In class amke_graph function used the method Graph which actualy merge nodes array, adjacency matrix, target array, all that in one graph.Then dataset may be splitted, shuffled and loaded into so called "loader" (special object for graph neural network).
- Model creating stage begin with definig of two function which needs in layers with random distributed weights. Next - definition of Net class with layers initialising in \_\_init__ function and layers usage in call function. Here you can implement or delete the layers, change parameters inside layers, add regularisation and activation (corresponding documentation contain all layers options).
- Training stage begin with creating the object of Net class, choosing the optimizer method, loss and accuracy functions. Then you can see custom training loop with realisation of gradient descent and data presenting for each epoch.

 

#### Second

Conceptually the same as previous but with one significant difference. Here we use as loss function so called [CategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy). For that purpose our [json_creator_2](gnn_with_prob/2_add_crossentropy/json_creator_2.py) did the target data of [data_global_2](gnn_with_prob/2_add_crossentropy/data_global_2.json) in the form of set of probabilities for numbers of interactions insted of scalar meaning:  
<img src="https://render.githubusercontent.com/render/math?math=\vec{Y}=[P_{1},P_{2},...,P_{8}]">  
In case of input data only one probability should be equal to 1, others - 0 and for computed output data is more distributed content.

#### Third

Previous two model both used the same [convolution layers](https://graphneural.network/layers/convolution/#generalconv) whose input data is feature fector and adjacencymatrix as input. In third model we added edges matrix - extra information about connections beatween nodes, and therefore we use another [convolution layers](https://graphneural.network/layers/convolution/#crystalconv). It is the essential difference of third model. Of course [json_creator](gnn_with_prob/3_add_directive_vector_in_graph/json_creator_3.py) also changed to give [output file](gnn_with_prob/3_add_directive_vector_in_graph/data_global_3.json) with edges matrix (E).

### Result
...

## Pulse shapes as member of input data (secondary approach)

In directory [nn_with_ps](nn_with_ps) you can find set of codes which was created to extract batches of pulse shapes which occured at the same time range. Also with respect to every signals it counts number of interactions. Resulting data you can find in [json file](nn_with_ps/data.json).



How it works:
1. By the [path](nn_with_ps/simpulses1/simpulses/Data) we have data files for each crystal about charge-carriers in different segments. All the information in file splited by time ranges. With this information we can create the pulse shape. 
    Speciall for those files was created [code](nn_with_ps/take_time_of_bit.py) to find what bit of data correspond to knowne time.
2. From [Only_gamma_events.0000](nn_with_ps/simpulses1/simpulses/OnlyGammaEvents.0000) we can count number of interactions correspond to crystal, segment and time range. With this information and previouse point it is possible to [build the mask](nn_with_ps/read_only_gamma.py): \[bit, crystal, segment, number of interactions].
3. With mask we know in which crystal interaction occurred so we can use [special code](nn_with_ps/simpulses1/eventstoascii.cpp) (not mine) which product the pulse shapes. Than knowing the number of data bit and the segment we can extract proper to number of interactions the [pulse shape](nn_with_ps/cut_ps_by_num.py) and [build the dataset](nn_with_ps/build_the_dataset.py).
4. [To create the json file](nn_with_ps/create_json.py) with graphs the time of interaction was added to dataset. So in loop we can find out which interactions occurred at the same time range and merge thay in one graph.


