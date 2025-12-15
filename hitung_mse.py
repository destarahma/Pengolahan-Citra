import cv2
import numpy as np
import pandas as pd

# ======================
# Load citra grayscale
# ======================
img = cv2.imread("hasil_landscape_grayscale.png", cv2.IMREAD_GRAYSCALE)

# ======================
# Fungsi Noise
# ======================
def salt_pepper(img, amount=0.05):
    noisy = img.copy()
    num = int(amount * img.size)

    coords = [np.random.randint(0, i - 1, num) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i - 1, num) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def gaussian_noise(img):
    gauss = np.random.normal(0, 25, img.shape)
    noisy = img + gauss
    return np.uint8(np.clip(noisy, 0, 255))

# ======================
# Fungsi Filter
# ======================
def mean_filter(img):
    return cv2.blur(img, (3,3))

def median_filter(img):
    return cv2.medianBlur(img, 3)

# ======================
# Fungsi MSE
# ======================
def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

# ======================
# Proses
# ======================
sp_noise = salt_pepper(img)
gauss_noise = gaussian_noise(img)

sp_median = median_filter(sp_noise)
sp_mean = mean_filter(sp_noise)

# ======================
# Tabel Hasil MSE
# ======================
data = {
    "Perbandingan": [
        "Grayscale vs Salt & Pepper",
        "Grayscale vs Gaussian",
        "Grayscale vs SP + Median Filter",
        "Grayscale vs SP + Mean Filter"
    ],
    "Nilai MSE": [
        mse(img, sp_noise),
        mse(img, gauss_noise),
        mse(img, sp_median),
        mse(img, sp_mean)
    ]
}

df = pd.DataFrame(data)
print(df)
