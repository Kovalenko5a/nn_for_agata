import numpy as np
from tensorflow import keras as kr
from keras.layers.core import Dense
import matplotlib.pyplot as plt
import pandas as pd
from for_4_conv_data import global_array


gl_ary = global_array()
feature_array = gl_ary[:2]

#matrix of numbers of interactions
lable_array = gl_ary[2]

#feature_array = np.reshape(feature_array, (feature_array.shape[1], feature_array.shape[2],feature_array.shape[3],feature_array.shape[0]))

#matrix of edep
tr_features1 = feature_array[0]

#matrix of number of crystal (same information in shape of matrix, but try)
tr_features2 = feature_array[1]

feature_shape = tr_features1.shape
lable_shape = lable_array.shape



#model
## this model has two input dataset which merge (Concatenate) before output + there are Pooling and Reshaping experement.
### (if comment one of layer please check that shape is proper)
def create_model(learning_rate):
    input1 = kr.layers.Input(shape=(feature_shape[1],feature_shape[2],), name="one-hot")
#     input1 = kr.layers.Input(input_shape=feature_shape)
    x1 = kr.layers.MaxPooling1D(2)(input1)
    x1 = kr.layers.Dense(60, activation="sigmoid")(x1)
    x1 = kr.layers.Permute((2,1))(x1)
    x1 = kr.layers.MaxPooling1D(2)(x1)
    x1 = kr.layers.Dense(56, activation="sigmoid")(x1)
    x1 = kr.layers.Permute((2,1))(x1)
    x1 = kr.layers.Dense(165, activation="sigmoid")(x1)
    x1 = kr.layers.Permute((2,1))(x1)
    ############################
    input2 = kr.layers.Input(shape=(feature_shape[1],feature_shape[2],), name="edep-crys")
    x2 = kr.layers.MaxPooling1D(2)(input2)
#     x2 = kr.layers.Dense(60, activation="sigmoid")(x2)
#     x2 = kr.layers.Permute((2,1))(x2)
#     x2 = kr.layers.MaxPooling1D(2)(x2)
#     x2 = kr.layers.Dense(56, activation="sigmoid")(x2)
#     x2 = kr.layers.Permute((2,1))(x2)
    x2 = kr.layers.Dense(165, activation="sigmoid")(x2)
    x2 = kr.layers.Permute((2,1))(x2)
    x2 = kr.layers.Dense(56, activation="softmax")(x2)
    ############################
    concatted = kr.layers.Concatenate()([x1, x2])
    #concatted = kr.layers.Dense(56)(concatted)
    #concatted = kr.layers.Dense(4, activation="sigmoid")(concatted)
    out = kr.layers.Dense(56, activation="softmax")(concatted)
    model = kr.Model(inputs=[input1, input2], outputs = out, name="...")
    opt=kr.optimizers.RMSprop(learning_rate=learning_rate)
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
learning_rate=0.05
epochs=25
batch_size=4

# Create and compile the model's topography.
my_model = create_model(learning_rate)

#train model
epochs, rmse = train_model(my_model, epochs, batch_size, {"one-hot": tr_features1, "edep-crys": tr_features2}, lable_array)

#ploting the accuracy curve
plot_the_accuracy_curve(epochs, rmse)

##for testing, but need to create test data sets
# metrics = my_model.evaluate(x={"one-hot": te_features1, "edep-crys": te_features2}, y=te_labels, batch_size=batch_size,  verbose=1)
# print("%s: %.2f%%" % (my_model.metrics_names[1], metrics[1]*100))
