import numpy as np
import math
import tensorflow as tf
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import seaborn as sbn
import tensorflow_probability as tfp
import scipy.sparse as sp


from for_conv_graph_2 import global_array
X, Y, A = global_array()

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
def expon(x, ampl, decr):
    return ampl*np.exp(-decr*x)
yy = []
for i in Y:
    for j in i:
        yy.append(j)
xxx = np.array(range(10))
counts = np.bincount(yy)
popt, _ = curve_fit(expon, xxx[1:], counts[1:]/counts.sum())
XX = np.array([0.0001*i for i in range(10000,90001)])
##########
# plt.plot(XX, expon(XX, *popt), color='green')
# plt.plot(xxx[1:], counts[1:]/counts.sum(), "*")


# a = tfp.distributions.Exponential(rate=popt[1])
# b = popt[0]*a.sample(sample_shape=([len(yy)]))
# #bb = tf.round(b)
# sbn.distplot(x=b, hist=True)
##########
popt1, _1 = curve_fit(expon, xxx[1:], counts[1:])
plt.plot(XX, expon(XX, *popt1), color='green')
plt.plot(xxx[1:], counts[1:], "*")

a1 = tfp.distributions.FiniteDiscrete(xxx[1:], probs=counts[1:]/counts.sum())
b1 = a1.sample(sample_shape=([len(yy)]))
sbn.histplot(x=b1)