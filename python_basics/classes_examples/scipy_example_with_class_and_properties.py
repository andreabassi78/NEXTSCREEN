import matplotlib.pyplot as plt
from scipy import ndimage, datasets

class Image:

    def __init__(self, data, is_human_data=False):
        self.data = data
        self.is_human_data = is_human_data
        self._shape = data.shape  # Private attribute
        
    @property
    def shape(self):
        '''You can put anything here
        even "import os; os.system('rm -rf /')"
        don't do it!
        '''
        print("And the shape is:", self._shape)
        return self._shape

    # Setter for shape (also reshapes the image)
    @shape.setter
    def shape(self, new_shape):
        if not isinstance(new_shape, tuple) or len(new_shape) != 2:
            raise ValueError("Shape must be a tuple of (height, width).")
        old_shape = self._shape
        zoom = [new_shape[0]/old_shape[0], new_shape[1]/old_shape[1]]
        self._rescale(zoom)
        print("Reshaped to:", my_first_image._shape)
        
    def _rescale(self, zoom=[0.5,0.5]):
        self.data = ndimage.zoom(self.data, zoom)
        self._shape = self.data.shape  # Automatically update shape

    def filter(self, sigma=2):
        self.data = ndimage.gaussian_filter(self.data, sigma)

    def __add__(self, other):
        if self.shape != other.shape:
            other.shape = self.shape  # Reshape other image instead of modifying permanently
        result = self._data + other._data
        is_result_human_data = self._is_human_data or other._is_human_data
        return Image(data=result, is_human_data=is_result_human_data)

if __name__ == '__main__':

    my_first_image = Image(data=datasets.ascent())
    my_first_image.shape
    my_first_image.shape = (200, 200)  # Set the property, triggers reshaping of the image

    plt.imshow(my_first_image.data)
    plt.colorbar()
    plt.show()