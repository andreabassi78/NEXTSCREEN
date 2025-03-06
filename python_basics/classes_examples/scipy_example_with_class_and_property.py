import matplotlib.pyplot as plt
from scipy import ndimage, datasets

class Image:

    def __init__(self, data, is_human_data=False):
        self.data = data
        self.is_human_data = is_human_data
        self._shape = data.shape # Private attribute

    @property
    def shape(self):
        """You can put anything you want here
        """
        print("And the shape is:", self._shape)
        return self._shape
    
    @shape.setter
    def shape(self, new_shape):
        if not isinstance(new_shape,list) or len(new_shape) !=2:
            raise ValueError("Shape must be a tuble of (height, width).")
        old_shape = self._shape
        zoom = [new_shape[0]/old_shape[0], new_shape[1]/old_shape[1]]
        self.rescale(zoom)
        print('Reshaped to', self._shape)

    def filter(self, sigma=2):
        self.data = ndimage.gaussian_filter(self.data, sigma)
    
    def rescale(self, zoom=0.5):
        self.data = ndimage.zoom(self.data, zoom)
        self._shape = self.data.shape

    def __add__(self, other):
        if self.shape != other.shape:
            zoom = [self._shape[0]/other._shape[0],self._shape[1]/other._shape[1]]
            other.rescale(zoom) # very dangerous because we are permanently changing the size of other 
        result = self.data + other.data
        is_result_human_data = (self.is_human_data or other.is_human_data) 
        return Image(data=result, is_human_data=is_result_human_data)

if __name__ == '__main__':

    my_first_image = Image(data=datasets.ascent())
    my_first_image.shape # using the getter function
    my_first_image.shape = [256,256] # using the setter function

    plt.imshow(my_first_image.data)
    plt.colorbar()
    plt.show()
