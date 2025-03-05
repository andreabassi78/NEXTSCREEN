import matplotlib.pyplot as plt

from scipy import ndimage, datasets

data = datasets.ascent()

sigma = 3

data = ndimage.gaussian_filter(data,sigma)

zoom = 0.5        
data = ndimage.zoom(data, zoom) 
shape = data.shape

print(shape)

plt.imshow(data)
plt.colorbar()
plt.show()