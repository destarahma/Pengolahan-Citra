# edge_detection_demo.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def roberts_edge(image):
    # Roberts Cross kernels
    kernel_x = np.array([[ 1,  0],
                         [ 0, -1]], dtype=int)
    kernel_y = np.array([[ 0,  1],
                         [-1,  0]], dtype=int)
    gx = ndimage.convolve(image.astype(int), kernel_x)
    gy = ndimage.convolve(image.astype(int), kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def prewitt_edge(image):
    # Prewitt kernels
    kernel_x = np.array([[ 1,  0, -1],
                         [ 1,  0, -1],
                         [ 1,  0, -1]], dtype=int)
    kernel_y = np.array([[ 1,  1,  1],
                         [ 0,  0,  0],
                         [-1, -1, -1]], dtype=int)
    gx = ndimage.convolve(image.astype(int), kernel_x)
    gy = ndimage.convolve(image.astype(int), kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def sobel_edge(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def frei_chen_edge(image):
    # define Frei-Chen kernels (W1..W4 for edges)
    sqrt2 = np.sqrt(2)
    W1 = np.array([[ 1,  sqrt2,  1],
                   [ 0,       0,  0],
                   [-1, -sqrt2, -1]], dtype=float)
    W2 = np.array([[ 1,   0,  -1],
                   [ sqrt2, 0, -sqrt2],
                   [ 1,   0,  -1]], dtype=float)
    # optionally W3, W4 etc. but we only use two main for horizontal/vertical-ish edges
    gx = ndimage.convolve(image.astype(float), W1)
    gy = ndimage.convolve(image.astype(float), W2)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255))

if __name__ == "__main__":
    img = cv2.imread(r"C:\VScode\citra kelompok\hasil_landscape_grayscale.png"
, cv2.IMREAD_GRAYSCALE)

    edges = {
        "Original": img,
        "Roberts": roberts_edge(img),
        "Prewitt": prewitt_edge(img),
        "Sobel": sobel_edge(img),
        "Frei-Chen": frei_chen_edge(img),
    }

    plt.figure(figsize=(15, 8))
    for i, (name, e) in enumerate(edges.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(e, cmap='gray')
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
