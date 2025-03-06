import numpy as np

a  = np.random.random([3,2])

print('Shape of the array a:', a.shape) # use a getter function of property shape

a.shape = [2,3] # use a setter function of the property shape

print('Shape of the array a, now', a.shape)