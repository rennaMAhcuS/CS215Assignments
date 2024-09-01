import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'Cambria'

img1 = plt.imread('Mona_Lisa.jpg')
if img1.dtype == 'uint8':
    img1 = img1.astype(float)

shifted_images = []
tx_values = list(range(-10, 0)) + list(range(1, 11))
height, width = img1.shape[:2]

for tx in tx_values:
    shifted_img = np.zeros_like(img1)
    if tx > 0:
        shifted_img[:, tx:] = img1[:, :-tx]
    elif tx < 0:
        shifted_img[:, :width + tx] = img1[:, -tx:]
    shifted_images.append(shifted_img)

# Plot the results
fig, axes = plt.subplots(4, 5, figsize=(15, 10), dpi=300)
axes = axes.flatten()

for ax, img, tx in zip(axes, shifted_images, tx_values):
    ax.imshow(img.astype(np.uint8))
    ax.set_title(f'Shift tx={tx}')
    ax.axis('off')

plt.tight_layout()

# plt.show()

plt.savefig("output/shift_monalisa.png")

org_img = np.mean(img1, axis=2)
org_img = org_img.flatten()

corr_coefficient_list = []
tx_values_with_zero = list(range(-10, 11))
for tx in tx_values_with_zero:
    if tx == 0:
        test_img = org_img
    else:
        test_img = np.mean(shifted_images[tx_values.index(tx)], axis=2)
        test_img = test_img.flatten()

    corr_coefficient, _ = pearsonr(org_img, test_img)
    corr_coefficient_list.append(corr_coefficient)

X = np.array(tx_values_with_zero)
plt.figure(dpi=300)
plt.plot(X, corr_coefficient_list)
custom_ticks = range(-10, 11, 2)
plt.xticks(custom_ticks)
plt.xlabel('Shift Values')
plt.ylabel('Correlation Coefficients')

# plt.show()

plt.savefig("output/correlation_coefficient.png")

image = Image.open('Mona_Lisa.jpg').convert('L')
image_array = np.array(image)

flattened_array = image_array.flatten()
num_bins = 256

hist = np.zeros(num_bins, dtype=int)
for pixel in flattened_array:
    hist[pixel] += 1

total_pixels = flattened_array.size
normalized_hist = hist / total_pixels

plt.figure(figsize=(12, 6), dpi=300)

image = plt.imread('Mona_Lisa.jpg')
rgb_array = np.array(image)

r_channel = rgb_array[:, :, 0]
g_channel = rgb_array[:, :, 1]
b_channel = rgb_array[:, :, 2]

r_flat = r_channel.flatten()
g_flat = g_channel.flatten()
b_flat = b_channel.flatten()

def calculate_histogram(flat_array, calc_hist_num_bins=256):
    calc_hist_hist = np.zeros(calc_hist_num_bins, dtype=int)
    for calc_hist_pixel in flat_array:
        calc_hist_hist[calc_hist_pixel] += 1
    return calc_hist_hist

hist_r = calculate_histogram(r_flat)
hist_g = calculate_histogram(g_flat)
hist_b = calculate_histogram(b_flat)

total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
normalized_hist_r = hist_r / total_pixels
normalized_hist_g = hist_g / total_pixels
normalized_hist_b = hist_b / total_pixels

plt.subplot(2, 2, 1)
plt.plot(normalized_hist_r, color='red', label='Normalized R Histogram')
plt.fill_between(range(num_bins), normalized_hist_r, color='red', alpha=0.3)
plt.legend()
plt.title('Red Channel')

plt.subplot(2, 2, 2)
plt.plot(normalized_hist_g, color='green', label='Normalized G Histogram')
plt.fill_between(range(num_bins), normalized_hist_g, color='green', alpha=0.3)
plt.legend()
plt.title('Green Channel')

plt.subplot(2, 2, 3)
plt.plot(normalized_hist_b, color='blue', label='Normalized B Histogram')
plt.fill_between(range(num_bins), normalized_hist_b, color='blue', alpha=0.3)
plt.legend()
plt.title('Blue Channel')

plt.subplot(2, 2, 4)
plt.plot(normalized_hist, color='gray', label='Normalized Grayscale Histogram')
plt.fill_between(range(num_bins), normalized_hist, color='gray', alpha=0.3)
plt.legend()
plt.title('Gray Channel')

plt.tight_layout()

# plt.show()

plt.savefig("output/RGB_grayscale_channels_hist.png")