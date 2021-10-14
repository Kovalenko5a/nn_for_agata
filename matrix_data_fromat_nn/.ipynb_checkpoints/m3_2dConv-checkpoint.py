import numpy as np
from tensorflow import keras as kr
from keras.layers.core import Dense
import matplotlib.pyplot as plt
import pandas as pd
from for_3_conv_data import global_array

#global array which containe all matrices
gl_ary = global_array()


#matrix of edep
feature_array = gl_ary[0]
#matrix of number of interactions
lable_array = gl_ary[1]


#shape we use in model
feature_shape = feature_array.shape
lable_shape = lable_array.shape



#model
## there are Pooling and Reshaping experement (if comment one of layer please check that shape is proper).
def create_model(learning_rate):
    input1 = kr.layers.Input(shape=(feature_shape[1],feature_shape[2],))
#     input1 = kr.layers.Input(input_shape=feature_shape)
    x1 = kr.layers.MaxPooling1D(2)(input1)
    x1 = kr.layers.Dense(60, activation="sigmoid")(x1)
    x1 = kr.layers.Permute((2,1))(x1)
    x1 = kr.layers.MaxPooling1D(2)(input1)
    x1 = kr.layers.Dense(56, activation="sigmoid")(x1)
    x1 = kr.layers.Permute((2,1))(x1)
    x1 = kr.layers.Dense(165, activation="sigmoid")(x1)
    x1 = kr.layers.Permute((2,1))(x1)
    #x1 = kr.layers.Dense(40, activation="sigmoid", kernel_regularizer=kr.regularizers.l2(l=0.1))(x1)
    ############################
    out = kr.layers.Dense(56, kernel_regularizer=kr.regularizers.l2(l=0.1), activation="sigmoid")(x1)
    model = kr.Model(inputs=input1, outputs=out, name="...")
    
    #choose the optimizer
    opt=kr.optimizers.Adam(learning_rate=learning_rate)
    #compile the model
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size, feature, lable):
    #run the standart training loop
    history = model.fit(x=feature, y=lable, epochs=epochs, batch_size=batch_size, verbose=1) #!!!!!
    
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
  
    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["accuracy"]

    return epochs, rmse

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
epochs=25
batch_size=3

# Create and compile the model's topography.
my_model = create_model(learning_rate)

#train model
epochs, rmse = train_model(my_model, epochs, batch_size, feature_array, lable_array)

#ploting the accuracy curve
plot_the_accuracy_curve(epochs, rmse)


#for testing, but need to create test data sets
# metrics = my_model.evaluate(x={"one-hot": te_features1, "edep-crys": te_features2}, y=te_labels, batch_size=batch_size,  verbose=1)
# print("%s: %.2f%%" % (my_model.metrics_names[1], metrics[1]*100))
