import matplotlib.pyplot as plt
from scipy import ndimage, datasets
import numpy as np

class Image:

    def __init__(self, data, is_human_data=False):
        self.data = data
        self.is_human_data = is_human_data
        self.shape = data.shape
    
    def filter(self, sigma=2):
        self.data = ndimage.gaussian_filter(self.data, sigma)
    
    def rescale(self, zoom=0.5):
        self.data = ndimage.zoom(self.data, zoom)
        self.shape = self.data.shape

    def __add__(self, other):
        if self.shape != other.shape:
            zoom = [self.shape[0]/other.shape[0],self.shape[1]/other.shape[1]]
            other.rescale(zoom) # very dangerous because we are permanently changing the size of other 
        result = self.data + other.data
        is_result_human_data = (self.is_human_data or other.is_human_data) 
        return Image(data=result, is_human_data=is_result_human_data)

if __name__ == '__main__':

    my_first_image = Image(data=datasets.ascent())
    my_first_image.rescale(0.5)
    # my_first_image.filter(sigma = 3)

    my_second_image = Image(data = np.random.uniform(30.0,120.0,[512,512]))
    
    # result_image = my_first_image.__add__(my_second_image)
    result_image = my_first_image + my_second_image

    print(my_second_image.shape) # the shape was changed when the __add__ method was executed

    plt.imshow(result_image.data)
    plt.colorbar()
    plt.show()

    #print(dir(Image))