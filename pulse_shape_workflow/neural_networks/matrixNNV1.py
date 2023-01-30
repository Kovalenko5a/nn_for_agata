import numpy as np
import math
import json
import tensorflow as tf
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from Conv4D import Conv4D



import tensorflow_probability as tfp

import scipy.sparse as sp

from tensorflow import keras as kr
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.losses import MeanAbsoluteError, Loss, MeanSquaredError
from tensorflow.keras.metrics import AUC, Accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


import matplotlib.pyplot as plt
import pandas as pd


import seaborn as sbn

globalArrayFeatureData = []
globalArrayLableData = []

for numOfTG in range(100):
    TGdataSnapshot = np.load('../compresed_dataset/gateTimeData'+str(numOfTG)+'.npz')
    globalArrayFeatureData.append(TGdataSnapshot['arr_0'][:50])
    globalArrayLableData.append([TGdataSnapshot['arr_0'][50]])
    if numOfTG%100==0:
        print(numOfTG)

globalArrayFeatureDataTF = tf.convert_to_tensor(globalArrayFeatureData, dtype=float)
globalArrayLableDataTF = tf.convert_to_tensor(globalArrayLableData, dtype=float)


del globalArrayFeatureData, globalArrayLableData


#batch size is 30 currently  - the number of time gates we go through
inputShape = tf.shape(globalArrayFeatureDataTF)
outputShape = tf.shape(globalArrayLableDataTF)
print(inputShape)
print(outputShape)

conv4d_1 = Conv4D(output_channels=1, kernel_shape=(1, 1, 1, 1), padding='valid')
dense = tf.keras.layers.Dense(6)
permute = kr.layers.Permute((2,1))
out = conv4d_1(globalArrayFeatureDataTF)
out2 = dense(globalArrayFeatureDataTF)
out3 = permute(globalArrayFeatureDataTF)

print(tf.shape(out2))
print(tf.shape(out3))
breakpoint()

#model
## this model has two input dataset which merge (Concatenate) before output + there are Pooling and Reshaping experement.
### (if comment one of layer please check that shape is proper): https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
def create_model(learning_rate):
    input1 = kr.layers.Input(shape=inputShape[1:], name="one-hot")
    out = Conv4D(output_channels=1, kernel_shape=(1, 1, 1, 1), padding='valid')(input1)
    model = kr.Model(inputs= input1, outputs = out, name="...")
    # opt=kr.optimizers.RMSprop(learning_rate=learning_rate)
    opt=kr.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])
    return model

#define the train function
def train_model(model, epochs, batch_size, feature, lable):
    #run the standart training loop
    history = model.fit(x=feature, y=lable, epochs=epochs, batch_size=batch_size, verbose=1) #!!!!!
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
  
    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["accuracy"]

    return epochs, rmse

#define the ploting function
def plot_the_accuracy_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")

    plt.plot(epochs, rmse, label="'||'")
    plt.legend()
    plt.ylim([rmse.min()*0.94, rmse.max()* 1.05])
    plt.show()  

#hyperparameters
learning_rate=0.005
epochs=30
batch_size=100

# Create and compile the model's topography.
my_model = create_model(learning_rate)

#train model
epochs, rmse = train_model(my_model, epochs, batch_size, globalArrayFeatureDataTF , globalArrayLableDataTF)

#ploting the accuracy curve
plot_the_accuracy_curve(epochs, rmse)

##for testing, but need to create test data sets
# metrics = my_model.evaluate(x={"one-hot": te_features1, "edep-crys": te_features2}, y=te_labels, batch_size=batch_size,  verbose=1)
# print("%s: %.2f%%" % (my_model.metrics_names[1], metrics[1]*100))
# globalArrayFeatureData