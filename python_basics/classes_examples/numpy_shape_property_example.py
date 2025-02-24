import numpy as np

a = np.random.random([3,2])

print(a)
print(dir(a)) # show all the attributes and methods of the object a
print('Shape of array a:',a.shape) #use the getter

a.shape = [2,3]

print('Shape of array a:',a.shape) #use the setter, which triggers a reshape
