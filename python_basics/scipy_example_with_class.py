import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, datasets

class Image:

    def __init__(self, data, is_human_data=False):
        self.data = data
        self.is_human_data = is_human_data
        self.shape = data.shape

    def filter(self, sigma=2):
        self.data = ndimage.gaussian_filter(self.data,sigma)
        
    def rescale(self, zoom=0.5):
        self.data = ndimage.zoom(self.data, zoom) 
        self.shape = self.data.shape

    def __add__(self,other):
        if self.shape != other.shape:
            zoom = [self.shape[0]/other.shape[0],self.shape[1]/other.shape[1]] 
            other.rescale(zoom)    
        result=self.data+other.data
        return Image(result)

if __name__ == '__main__':

    print(dir(Image)) # print the methods and attributes of the class

    my_first_image = Image(data=datasets.ascent())
    my_first_image.rescale(2)
    my_first_image.filter(2)

    my_second_image = Image(data = np.random.uniform(30.0, 120.0,[512,512]))

    result_image = my_first_image+my_second_image

    plt.imshow(result_image.data)
    plt.colorbar()
    plt.show()