import numpy as np

a = np.array([[0,1,2,3],
              [1,2,3,4],
              [2,3,4,5]])
b = np.array([2,3,4])

# let's multiply b to each column of a

# method 1, never use this method
c = np.zeros_like(a)
columns_number = c.shape[1] 
for idx in range(columns_number):
    c[:,idx] = a[:,idx]*b
print('method1:',c)


# method 2, repeat b to obtain a matrix with size a, not efficient memory-wise
rows_number = c.shape[0] 
columns_number = c.shape[1] 
b2 = np.tile(b,[1,4]).reshape([4,3]).T # tile is the equivalent to reshape in Matlab
c = a*b2
print(c)

# method 3, unconvenient double transpose
c = (a.T*b).T
print(c)

# method 4, "broadcasting", computationally efficient, works in n dimensions
c = a*b[:,None]
print(c)

# method 4, elegant "broadcasting", computationally efficient, works in n dimensions
# https://numpy.org/doc/stable/user/basics.broadcasting.html
c = a*b[:,np.newaxis] # newaxis expand the array adding a new empty dimension
print(c)
print(b[:,np.newaxis].shape)

# method 5, Eistein sum, only for the brave
c = np.einsum('ij,i->ij', a, b)
print(c) 
# einsum is the tool of choice for tensors operations
# 'ij,i->ij' tells einsum to:
# Take the elements of a (indexed by ij).
# Multiply them with the elements of b (indexed by i).
# Return the resulting array with the same shape as a.