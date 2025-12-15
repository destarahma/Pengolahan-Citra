import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def roberts_edge(image):
    kernel_x = np.array([[1, 0],
                         [0, -1]])
    kernel_y = np.array([[0, 1],
                         [-1, 0]])
    gx = ndimage.convolve(image.astype(int), kernel_x)
    gy = ndimage.convolve(image.astype(int), kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(magnitude, 0, 255))

# ==== PATH GAMBAR (ISI SENDIRI) ====
img = cv2.imread(r"C:\VScode\citra kelompok\gambar.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: Gambar tidak ditemukan!")
    exit()

edges = roberts_edge(img)

plt.imshow(edges, cmap='gray')
plt.title("Roberts Edge")
plt.axis('off')
plt.show()
