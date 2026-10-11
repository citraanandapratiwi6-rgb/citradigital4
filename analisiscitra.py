import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Baca citra hasil crop (ganti 'nomor_crop.png' sesuai nama file Anda)
img = cv2.imread('09_CombinedDegradation.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Penerapan Metode Enhancement
# A. Brightness Adjustment (+40)
brightness_img = cv2.add(img, 40)

# B. Contrast Stretching
min_val, max_val = np.min(img), np.max(img)
contrast_img = ((img - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

# C. Histogram Equalization
equalized_img = cv2.equalizeHist(img)

# 3. Visualisasi Perbandingan Citra dan Histogram
titles = [
    'Citra Asli',
    'Brightness Adjustment',
    'Contrast Stretching',
    'Histogram Equalization',
]
images = [img, brightness_img, contrast_img, equalized_img]

fig, axes = plt.subplots(4, 2, figsize=(12, 10))

for i in range(4):
    # Menampilkan Citra
    axes[i, 0].imshow(images[i], cmap='gray', vmin=0, vmax=255)
    axes[i, 0].set_title(f'Citra: {titles[i]}')
    axes[i, 0].axis('off')

    # Menampilkan Histogram
    axes[i, 1].hist(images[i].ravel(), bins=256, range=[0, 256], color='black')
    axes[i, 1].set_title(f'Histogram: {titles[i]}')
    axes[i, 1].set_xlim([0, 256])

plt.tight_layout()
plt.show()