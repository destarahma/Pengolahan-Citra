import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def prewitt_edge(image):
    kernel_x = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])
    gx = ndimage.convolve(image.astype(int), kernel_x)
    gy = ndimage.convolve(image.astype(int), kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(magnitude, 0, 255))

# ==== PATH GAMBAR (GANTI SENDIRI) ====
img = cv2.imread(r"C:\VScode\citra kelompok\gambar.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: Gambar tidak ditemukan!")
    exit()

edges = prewitt_edge(img)

plt.imshow(edges, cmap='gray')
plt.title("Prewitt Edge")
plt.axis('off')
plt.show()
