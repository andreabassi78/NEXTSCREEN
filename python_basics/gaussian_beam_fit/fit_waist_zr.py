import numpy as np
import tifffile     
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os  

# ----------------------- Parameters -----------------------
filename = 'beam_2_5_x.tiff'
camera_pixel = 4.8  # µm
M = 2.5             # magnification
wavelength = 0.473  # µm
n = 1               # refractive index

pixel_size = camera_pixel / M  # final pixel size in µm

# ----------------------- Load Image -----------------------
full_path = os.path.realpath(__file__)
folder, _ = os.path.split(full_path) 
file_path = os.path.join(folder, filename)

image = tifffile.imread(file_path)

# ----------------------- Crop Image Around Max -----------------------
crop_height = 100  # Half-height (vertical) in pixels
crop_width = 500   # Half-width (horizontal) in pixels

# Find beam peak
max_y, max_z = np.unravel_index(np.argmax(image), image.shape)

# Clamp crop to stay within image bounds
y1 = max(0, max_y - crop_height)
y2 = min(image.shape[0], max_y + crop_height)
z1 = max(0, max_z - crop_width)
z2 = min(image.shape[1], max_z + crop_width)

image_crop = image[y1:y2, z1:z2]

# ----------------------- Fit Function (µm) -----------------------
def gaussian_beam(yz, A, y0, z0, w, zr, offset):
    y, z = yz  # y and z in µm
    w_z = w * np.sqrt(1 + ((z - z0) / zr) ** 2)
    I = A * np.exp(-2 * (y - y0) ** 2 / w_z ** 2) * (w / w_z) ** 2 + offset
    return I.ravel()

# ----------------------- Prepare Grid in µm -----------------------
_y = np.arange(image_crop.shape[0]) * pixel_size
_z = np.arange(image_crop.shape[1]) * pixel_size
z, y = np.meshgrid(_z, _y)  # both in µm

y_flat = y.ravel()
z_flat = z.ravel()
image_flat = image_crop.ravel()

# ----------------------- Initial Parameters (µm) -----------------------
amplitude = np.max(image_crop) - np.min(image_crop)
y_center = (_y[-1] + _y[0]) / 2  # center in µm
z_center = (_z[-1] + _z[0]) / 2
w0 = 5 #um
zr0 = 100 #(_z[-1] - _z[0]) / 2       # ~1/2 of horizontal range in µm
offsets = np.min(image_crop)

p0 = [amplitude, y_center, z_center, w0, zr0, offsets]

# ----------------------- Bounds (in µm) -----------------------
bounds = (
    [0, _y[0], _z[0], 1e-2, 1e-2, 0],  # Lower bounds
    [
        np.inf,
        _y[-1],
        _z[-1],
        (_y[-1] - _y[0]) / 2,     # max waist = half vertical range
        (_z[-1] - _z[0]) * 2,     # max Rayleigh = 2x horizontal
        np.max(image_crop)
    ]
)

# ----------------------- Fit -----------------------
popt, _ = curve_fit(
    gaussian_beam,
    (y_flat, z_flat),
    image_flat,
    p0=p0,
    bounds=bounds,
    method='trf',
    maxfev=10000
)

# ----------------------- Generate Fitted Data -----------------------
data_fitted = gaussian_beam((y, z), *popt).reshape(image_crop.shape)

# ----------------------- Plot -----------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Original + contour (in µm)
axs[0].imshow(image_crop, cmap='jet', origin='lower',
              extent=[_z[0], _z[-1], _y[0], _y[-1]],
              aspect='auto')
axs[0].contour(z, y, data_fitted, 8, colors='w')
axs[0].set_title("Original Image with Fit Contours")
axs[0].set_xlabel("z (µm)")
axs[0].set_ylabel("y (µm)")

# Fitted data only (in µm)
im = axs[1].imshow(data_fitted, cmap='jet', origin='lower',
                   extent=[_z[0], _z[-1], _y[0], _y[-1]],
                   aspect='auto')
axs[1].set_title("Fitted Gaussian Beam Profile")
axs[1].set_xlabel("z (µm)")
axs[1].set_ylabel("y (µm)")

fig.colorbar(im, ax=axs[1], shrink=0.8)

plt.tight_layout()
plt.show()

# ----------------------- Output Parameters (in µm) -----------------------
A_fit, y0_fit, z0_fit, w_fit, zr_fit, offset_fit = popt
print(f"Fitted parameters (real units):")
print(f" - Waist w: {w_fit:.2f} µm")
print(f" - Rayleigh range zr: {zr_fit:.2f} µm")
print(f" - Beam center: y0 = {y0_fit:.2f} µm, z0 = {z0_fit:.2f} µm")
