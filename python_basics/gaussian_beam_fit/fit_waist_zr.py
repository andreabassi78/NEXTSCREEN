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

pixel_size = camera_pixel / M

# ----------------------- Load Image -----------------------
full_path = os.path.realpath(__file__)
folder, _ = os.path.split(full_path) 
file_path = os.path.join(folder, filename)

image = tifffile.imread(file_path)

# ----------------------- Crop Image Around Max -----------------------
crop_height = 100  # Half-height (vertical)
crop_width = 500   # Half-width (horizontal)

# Find beam peak
max_y, max_z = np.unravel_index(np.argmax(image), image.shape)

# Clamp crop to stay within image bounds
y1 = max(0, max_y - crop_height)
y2 = min(image.shape[0], max_y + crop_height)
z1 = max(0, max_z - crop_width)
z2 = min(image.shape[1], max_z + crop_width)

image_crop = image[y1:y2, z1:z2]

# ----------------------- Fit Function -----------------------
def gaussian_beam(yz, A, y0, z0, w, zr, offset):
    y, z = yz
    w_z = w * np.sqrt(1 + ((z - z0) / zr) ** 2)
    I = A * np.exp(-2 * (y - y0) ** 2 / w_z ** 2) * (w / w_z) ** 2 + offset
    return I.ravel()

# ----------------------- Prepare Grid -----------------------
_y = np.arange(image_crop.shape[0])
_z = np.arange(image_crop.shape[1])
z, y = np.meshgrid(_z, _y)

y_flat = y.ravel()
z_flat = z.ravel()
image_flat = image_crop.ravel()

# ----------------------- Initial Parameters -----------------------
amplitude = np.max(image_crop) - np.min(image_crop)
y_center = image_crop.shape[0] / 2
z_center = image_crop.shape[1] / 2
w0 = image_crop.shape[0] / 50  # 
zr0 = image_crop.shape[1] / 2  # 

offsets = np.min(image_crop)

p0 = [amplitude, y_center, z_center, w0, zr0, offsets]

# ----------------------- Bounds -----------------------
bounds = (
    [0, 0, 0, 1, 1, 0],  # Lower
    [
        np.inf,
        image_crop.shape[0],
        image_crop.shape[1],
        image_crop.shape[0] /2,
        image_crop.shape[1] *2,
        np.max(image_crop)
    ]  # Upper
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

# Original + contour
axs[0].imshow(image_crop, cmap='jet', origin='lower', aspect='auto')
axs[0].contour(z, y, data_fitted, 8, colors='w')
axs[0].set_title("Original Image with Fit Contours")
axs[0].set_xlabel("z (pixels)")
axs[0].set_ylabel("y (pixels)")

# Fitted data only
im = axs[1].imshow(image_crop-data_fitted, cmap='jet', origin='lower', aspect='auto')
axs[1].set_title("Fitted Gaussian Beam Profile")
axs[1].set_xlabel("z (pixels)")
axs[1].set_ylabel("y (pixels)")

# Add colorbar to second subplot
fig.colorbar(im, ax=axs[1], shrink=0.8)

plt.tight_layout()
plt.show()

# ----------------------- Output Parameters -----------------------
A_fit, y0_fit, z0_fit, w_fit, zr_fit, offset_fit = popt
print(f"Fitted parameters:")
print(f" - Waist w (pixels): {w_fit:.2f} → {w_fit * pixel_size:.2f} µm")
print(f" - Rayleigh range zr (pixels): {zr_fit:.2f} → {zr_fit * pixel_size:.2f} µm")
print(f" - Beam center (y0, z0): ({y0_fit:.1f}, {z0_fit:.1f})")
