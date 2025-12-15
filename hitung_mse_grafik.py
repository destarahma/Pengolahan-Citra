import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load citra
img = cv2.imread("hasil_landscape_grayscale.png", cv2.IMREAD_GRAYSCALE)

# Noise
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
    return np.uint8(np.clip(img + gauss, 0, 255))

# Filter
def mean_filter(img):
    return cv2.blur(img, (3,3))

def median_filter(img):
    return cv2.medianBlur(img, 3)

# MSE
def mse(a, b):
    return np.mean((a.astype(float) - b.astype(float)) ** 2)

# Proses
labels = [
    "Salt & Pepper",
    "Gaussian",
    "SP + Median Filter",
    "SP + Mean Filter"
]

values = [
    mse(img, salt_pepper(img)),
    mse(img, gaussian_noise(img)),
    mse(img, median_filter(salt_pepper(img))),
    mse(img, mean_filter(salt_pepper(img)))
]

# Grafik
plt.figure()
plt.bar(labels, values)
plt.xlabel("Metode")
plt.ylabel("Nilai MSE")
plt.title("Perbandingan Nilai MSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
