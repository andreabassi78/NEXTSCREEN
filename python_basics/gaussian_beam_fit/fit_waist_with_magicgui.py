import numpy as np
import tifffile     
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline   
from magicgui import magicgui
import pathlib
import os

def gaussian(x, A, x0, w, offset):
    return A * np.exp(-((x - x0) ** 2) / (w ** 2)) + offset

def _calculate_waist(file_path,camera_pixel,M,wavelenght,n,spline_constant):

    image = tifffile.imread(file_path)
    pixel_size = camera_pixel/M
    num_rows, num_cols = image.shape
    fwhm_values = np.zeros(num_cols)

    for col in range(num_cols):    

        profile = image[:, col]      
        y = np.arange(num_rows) 
        # initial parameter
        amplitude = np.max(profile) - np.min(profile) 
        center = np.amax(y)/2                    
        w0 = np.amax(y) / 50                 
        offsets = np.min(profile)                     
        p0 = [amplitude, center, w0, offsets]

        popt, _ = curve_fit(gaussian, y, profile, p0=p0)     
        _, _, w_fit, _ = popt   
        fwhm = 2.355/2 * w_fit   
        fwhm_values[col] = fwhm     

        if col==num_cols//2:
            plt.figure()
            plt.plot(y,profile, color='m')
            plt.plot(y,gaussian(y,*popt), color='k')

    x = np.arange(num_cols) 
    spl = UnivariateSpline(x, fwhm_values, s=spline_constant)
    fitted_fwhm_values = spl(x)

    #find the minimum FWHM value and its index
    min_index = np.argmin(fitted_fwhm_values)
    min_fwhm = fitted_fwhm_values[min_index]
    min_w = min_fwhm *2/2.355 

    zr = np.pi* min_w**2/wavelenght*n
    w_zr = (np.sqrt(2))* (min_fwhm)

    left_side = fitted_fwhm_values[:min_index]
    right_side = fitted_fwhm_values[min_index:] 

    closest_left = np.argmin(np.abs(left_side - w_zr))
    closest_left_value = fitted_fwhm_values[closest_left]
    zr_left = (-closest_left+min_index)*pixel_size

    closest_right = np.argmin(np.abs(right_side - w_zr)) + min_index
    closest_right_value = fitted_fwhm_values[closest_right]
    zr_right = (closest_right-min_index)*pixel_size

    print(f"The minimum FWHM value is {min_fwhm:.2f} um, the waist is {min_w:.2f} um, at column {min_index}")
    print(f"Teoretical Rayleight distance {zr:.2f} um: FWHM: {w_zr:.2f}")
    print(f"Rayleigh distance (left): {zr_left:.2f} um, FWHM: {closest_left_value:.2f} um")
    print(f"Rayleigh distance (right): {zr_right:.2f} um, FWHM: {closest_right_value:.2f} um")

    plt.figure()
    plt.plot(fwhm_values, color='m')
    plt.plot(spl(x))

    plt.figure()
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.axvline(x=min_index, color='c', linestyle='--', label='Min FWHM')
    plt.axvline(x=closest_left, color='m', linestyle='--', label='Closest Left')
    plt.axvline(x=closest_right, color='m', linestyle='--', label='Closest Right')
    plt.legend()

    plt.show()


filename = 'beam_2_5_x.tiff'
full_path = os.path.realpath(__file__)
folder, _ = os.path.split(full_path) 
FILE_PATH = os.path.join(folder,filename)
    
@magicgui(
    call_button="Calculate parameters"
)
def calculate_waist(
    wavelength: float = 0.473,
    M:float = 2.5, # magnification of the SLM, given by the 4f system
    camera_pixel: float = 4.8,
    n:float = 1.0,
    spline_constant: float =100,
    file_path = pathlib.Path(FILE_PATH)):

    _calculate_waist(file_path,camera_pixel,M,wavelength,n,spline_constant)

calculate_waist.show(run=True)

    