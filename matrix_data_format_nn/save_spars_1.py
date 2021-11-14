import scipy.sparse
import numpy as np
from for_3_conv_data import global_array

X, Y = global_array(20000)
sparse_x = scipy.sparse.csc_matrix(X[0])
sparse_y = scipy.sparse.csc_matrix(Y[0])
scipy.sparse.save_npz('sparse_x.npz', sparse_x)
scipy.sparse.save_npz('sparse_y.npz', sparse_y)
file_num = 1
for x,y in zip(X,Y):
    sparse_x = scipy.sparse.csc_matrix(x)
    sparse_y = scipy.sparse.csc_matrix(y)
    #print(sparse_x, "\n", sparse_y)
    #print('########################')
    name_x = "sparse_storage_1/sparse_x" + str(file_num) + ".npz"
    name_y = "sparse_storage_1/sparse_y" + str(file_num) + ".npz"
    scipy.sparse.save_npz(name_x, sparse_x)
    scipy.sparse.save_npz(name_y, sparse_y)
    file_num+=1
    print(file_num)
    
    