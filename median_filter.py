import os
import cv2
import numpy as np

# Folder input dan output
input_folder = "noise_output"
output_folder = "median_output"

os.makedirs(output_folder, exist_ok=True)

# ================================
#  MANUAL MEDIAN FILTER
# ================================

def median_filter(image, k=3):

    # Jika RGB → filter tiap channel
    if len(image.shape) == 3:
        channels = []
        for c in range(3):
            ch = manual_median_channel(image[:, :, c], k)
            channels.append(ch)
        return cv2.merge(channels)
    else:
        return manual_median_channel(image, k)


def manual_median_channel(channel, k=3):
    pad = k // 2
    h, w = channel.shape
    padded = np.pad(channel, pad, mode="edge")
    output = np.zeros_like(channel)

    for i in range(h):
        for j in range(w):
            window = padded[i : i + k, j : j + k].flatten()
            median_value = np.median(window)
            output[i, j] = median_value

    return output

# ================================
#  PROCESS ALL IMAGES
# ================================

print("Memulai proses median filter manual...")

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)

        if img is None:
            print("Gagal membaca:", filename)
            continue

        # Proses median filter manual window 3×3
        result = median_filter(img, k=3)

        save_path = os.path.join(output_folder, "MEDIAN" + filename)
        cv2.imwrite(save_path, result)

        print("Selesai:", filename)

print("SELESAI! Semua median filter manual telah diproses.")
