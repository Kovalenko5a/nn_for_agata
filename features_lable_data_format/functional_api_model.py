import numpy as np
from tensorflow import keras as kr
from keras.layers.core import Dense
import matplotlib.pyplot as plt
import pandas as pd


target_variable = "num_of_int"
#feature_variable = ["edep", "crystal", "slice_sect"]
hiden_layer_size = 167
train_test_split=0.7
#read data
# 2 1d arrays which containe energy deposition in X row (feachure) and number of interaction in Y row (lable)


# data = pd.read_csv("/home/yantuzemec/data_anal/geant4_data_anal/For_nn_final.csv")
data = pd.read_csv("out.csv")
data.slice_sect = data.slice_sect/55.
data.crystal = data.crystal/163.
#shaffle the dataset:
data = data.reindex(np.random.permutation(data.index))

#spliting data
mask = np.random.rand(len(data))<train_test_split
train_data=data[mask]
test_data=data[~mask]

tr_features1 = np.array(train_data.drop(["crystal", "slice_sect"], axis=1))
#tr_features1 = np.expand_dims(tr_features1, axis=0)
tr_features2 = np.array(train_data[["crystal", "slice_sect"]])
#tr_features2 = np.expand_dims(tr_features2, axis=0)
tr_labels = np.array(train_data[[target_variable]])
#tr_labels = np.expand_dims(tr_labels, axis=1)
# tr_labels = np.expand_dims(tr_labels, axis=0)
# tr_labels = np.expand_dims(tr_labels, axis=0)

te_features1 = np.array(test_data.drop(["crystal", "slice_sect"], axis=1))
# te_features1 = np.expand_dims(te_features1, axis=0)
te_features2 = np.array(test_data[["crystal", "slice_sect"]])
te_labels = np.array(test_data[[target_variable]])
# te_labels = np.expand_dims(te_labels, axis=1)
# te_labels = np.expand_dims(te_labels, axis=0)
# te_labels = np.expand_dims(te_labels, axis=0)


input_shape1=tr_features1.shape
input_shape2=tr_features2.shape


#model
def create_model(learning_rate):
    input1 = kr.layers.Input(shape=(input_shape1[1],), name="one-hot")
    #x1 = kr.layers.Conv1D(80, 3)(input1)
#     input1 = kr.layers.Conv1D(80, 3, input_shape=(input_shape1[1],1,))
   # x1 = kr.layers.Dense(60, activation="sigmoid")(x1)
    #x1 = kr.layers.Conv1D(30, 2)(x1)
    x1 = kr.layers.Dense(80)(input1)
    x1 = kr.layers.Dense(60)(x1)
    x1 = kr.layers.Dense(40)(x1)
    x1 = kr.layers.Dense(2, kernel_regularizer=kr.regularizers.l2(l=0.1))(x1)
    ############################
    input2 = kr.layers.Input(shape=(input_shape2[1],), name="edep-crys")
    x2 = kr.layers.Dense(3)(input2)
    x2 = kr.layers.Dense(3)(x2)
    x2 = kr.layers.Dense(2)(x2)
    ############################
    concatted = kr.layers.Concatenate()([x1, x2])
    concatted = kr.layers.Dense(4)(concatted)
    concatted = kr.layers.Dense(4)(concatted)
    out = kr.layers.Dense(1)(concatted)
    model = kr.Model(inputs=[input1, input2], outputs = out, name="...")
    opt=kr.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size, feature, lable):
    history = model.fit(x=feature, y=lable, epochs=epochs, batch_size=batch_size, verbose=1) #!!!!!
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
  
    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["accuracy"]

    return epochs, rmse

def plot_the_loss_curve(epochs, rmse):
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
epochs=10
batch_size=2

# Create and compile the model's topography.
my_model = create_model(learning_rate)

#train model
epochs, rmse = train_model(my_model, epochs, batch_size, {"one-hot": tr_features1, "edep-crys": tr_features2}, tr_labels)

#ploting the loss curve
plot_the_loss_curve(epochs, rmse)

#accuracy

metrics = my_model.evaluate(x={"one-hot": te_features1, "edep-crys": te_features2}, y=te_labels, batch_size=batch_size,  verbose=1)
print("%s: %.2f%%" % (my_model.metrics_names[1], metrics[1]*100))
