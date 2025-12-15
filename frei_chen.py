import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def frei_chen_edge(image):
    sqrt2 = np.sqrt(2)
    W1 = np.array([[1, sqrt2, 1],
                   [0,     0, 0],
                   [-1, -sqrt2, -1]])

    W2 = np.array([[1,     0, -1],
                   [sqrt2, 0, -sqrt2],
                   [1,     0, -1]])

    gx = ndimage.convolve(image.astype(float), W1)
    gy = ndimage.convolve(image.astype(float), W2)

    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = magnitude / magnitude.max() * 255
    return np.uint8(np.clip(magnitude, 0, 255))

# ==== PATH GAMBAR (GANTI SENDIRI) ====
img = cv2.imread(r"C:\VScode\citra kelompok\gambar.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: Gambar tidak ditemukan!")
    exit()

edges = frei_chen_edge(img)

plt.imshow(edges, cmap='gray')
plt.title("Frei-Chen Edge")
plt.axis('off')
plt.show()
