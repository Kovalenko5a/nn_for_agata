#If sparse target vector than tf.keras.losses.CategoricalCrossentropy
#If use integers target vector tf.keras.losses.SparseCategoricalCrossentropy

import numpy as np
import math
import json
import time
import tensorflow as tf
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()



import tensorflow_probability as tfp

import scipy.sparse as sp

from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool, GlobalSumPool, MessagePassing, TAGConv, GeneralConv, GatedGraphConv
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms import GCNFilter

#from for_gnn_with_crossentropy import global_array

import seaborn as sbn

################################################################################
# Config
################################################################################
learning_rate = 0.001  # Learning rate
epochs = 20  # Number of training epochs
es_patience = 30  # Patience for early stopping
batch_size = 500  # Batch size
#number_of_graphs = 10000 #How many graphs use to train

#read data from file (see for_gnn_with_tfp.py)
#uncoment if want to use algorythm from 0
# X, Y, A = global_array(number_of_graphs)

#read data from file (see create_the_json_datafiles.py) and .json file

# JSON file
json_file = open('data_global_2.json', "r")
 
# Reading from file
json_data = json.loads(json_file.read())
X, Y, A = json_data["Graphs"]["X"], json_data["Graphs"]["Y"], json_data["Graphs"]["A"]

number_of_graphs = len(X) #How many graphs in file

################################################################################
# Load data
################################################################################
class MyDataset(Dataset):
    def __init__(self, n_samples, **kwargs):
        self.n_samples = n_samples
        super().__init__(**kwargs)

    def read(self):
        def make_graph(j):
            a = sp.csr_matrix(A[j])
            #y = sp.csr_matrix(Y[j])
            return Graph(x=np.array(X[j], dtype = 'float64'), a=a, 
                        # y=y
                        y = np.array(Y[j], dtype = 'int64')
                        )

        # We must return a list of Graph objects
        return [make_graph(j) for j in range(self.n_samples)]


data = MyDataset(number_of_graphs, 
#                  transforms=NormalizeAdj()
                )


#data for train and test (in program used only whole dataset for training)
data_tr = data[0:int(len(data)*0.6)]
data_va = data[int(len(data)*0.6):int(len(data)*0.8)]
data_te = data[int(len(data)*0.8):len(data)]

#node-level - true for prediction for each segmen
loader_tr = DisjointLoader(data_tr, batch_size=batch_size)
loader_va = DisjointLoader(data_va, node_level=True, batch_size=batch_size)
loader_te = DisjointLoader(data_te, node_level=True, batch_size=batch_size)
loader_main = DisjointLoader(data, node_level=True, batch_size=batch_size)

##Create the layer with random weights
# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


##Neural Network: _init_ - declare the layers and call - declare the use of layers 
class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GatedGraphConv(3, 10)
        self.conv2 = GeneralConv(60)
        self.pool1 = TopKPool(ratio=0.5)
        self.dense1 = Dense(50, activation="swish")
        self.dense2 = Dense(40)
        self.dense3 = Dense(30, activation="swish")
        self.dense31 = Dense(20)
        self.dense32 = Dense(10, activation="swish")
        self.dense33 = Dense(30)
        self.dense34 = Dense(30, activation="swish")
        self.dense35 = Dense(30)
        self.dense36 = tfp.layers.DenseFlipout(30, activation=tf.nn.relu),
        self.dense4 = Dense(5)
        self.dense41 = Dense(2)
        self.dense42 = Dense(5)
        self.dense5 = Dense(9, activation="softmax", #kernel_regularizer=tf.keras.regularizers.l2(l=0.1)
                           )
        
        
        ## Use the layer with random weights 
        self.denseV = tfp.layers.DenseVariational(
            units=30,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            #kl_weight=1 / train_size,
            #activation="sigmoid",
        )
        
        ## Use of distributions as output layers
        self.Poisson = tfp.layers.IndependentPoisson(1)
        self.Normal = tfp.layers.IndependentNormal(1)
        
        ## Atempt to use exponential distr defined above
        self.Exp = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfp.distributions.Exponential(
        rate=t[..., 0]),
        convert_to_tensor_fn=lambda s: s.sample(1)
        )

    def call(self, inputs):
        x, a, i = inputs
        
        #################################
        #x1 = self.conv1([x, a])
        x1 = self.conv2([x, a])
        #x, i = self.global_pool([x, i])
        #x1, a1, i1 = self.pool1([x, a, i])
        x1 = self.dense1(x1)
        x1 = self.dense2(x1)
        x1 = self.dense3(x1)
        x1 = self.dense31(x1)
        x1 = self.dense32(x1)
        
        x1 = self.denseV(x1)
        
        x1 = self.dense4(x1)
        x1 = self.dense41(x1)
        x1 = self.dense5(x1)
#         y1 = a1.sample(x1.shape)
#         y1 = tf.constant(y1)
#         output = tf.add(x1,y1)
#         output = self.Poisson(x1)
#         output = tf.reshape(output, ([output.shape[1],1]))
#         output+= 1
#         output = abs(output)
        output = np.around(x1, 3)
        return output

    
##creat the model object
model = Net()
##Optimizer for back propagation
optimizer = Adam(learning_rate=learning_rate)

##loss function wtih use in back propagation
#loss_fn = SparseCategoricalCrossentropy()
loss_fn = CategoricalCrossentropy()

##To see the succes of model:
accuracy = categorical_accuracy   
results_acc = [] # print out the result in last column
results_loss=[]

##timing
my_start_time = time.time()
print(my_start_time)
##custom training loop without testing and evaluating

for i in range(epochs):
    feature, lable = loader_main.__next__()
    print("\n")
    y = model(feature, training=True)
#     for k in range(len(lable)):
#         #print(lable[k],"      ", y[k])
    with tf.GradientTape() as tape:
        predictions = model(feature, training=True)
        loss = loss_fn(lable, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # unconnected_gradients delete the warning
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(accuracy(lable, predictions))
    results_acc.append(acc)
    results_loss.append(loss)
    print("#############")
    print("ACCURACY:")
    print(float(acc))
    print("LOSS:")
    print(float(loss))
    print("#############")

my_end_time = time.time()
print(my_end_time)
my_time = my_end_time - my_start_time
print(my_time)    

#plot accuracy curve
from matplotlib import pyplot as plt
epoch_nums = range(1,epochs+1)
ra = np.array(results_acc, dtype=float)
rl = np.array(results_loss, dtype=float)
plt.plot(epoch_nums, ra)
plt.legend('accuracy', loc='upper right')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(epoch_nums, rl)
plt.legend('loss', loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#clear RAM
del data_tr, data_va, data_te, loader_main, loader_te, loader_tr, loader_va