import cv2
import numpy as np
import os

# Folder input & output
input_folder = "noise_output"
output_folder = "mean_output"

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Manual Mean Filter 3x3
# -----------------------------
def manual_mean_filter(img):
    h, w = img.shape[:2]

    # Jika RGB → proses tiap channel
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = mean_kernel(img[:, :, c])
        return result
    else:
        return mean_kernel(img)

def mean_kernel(channel):
    h, w = channel.shape
    output = np.zeros_like(channel)

    # Padding 1 pixel (window 3x3)
    padded = np.pad(channel, pad_width=1, mode='edge')

    for i in range(h):
        for j in range(w):
            window = padded[i:i+3, j:j+3]

            # mean manual = jumlah pixel / 9
            output[i, j] = np.sum(window) // 9

    return output

# -----------------------------
# Eksekusi semua gambar
# -----------------------------
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_folder, filename)

        img = cv2.imread(path)

        print("Proses mean filter:", filename)
        filtered = manual_mean_filter(img)

        save_path = os.path.join(output_folder, "mean_" + filename)
        cv2.imwrite(save_path, filtered)

print("Selesai! Semua gambar telah difilter dan disimpan di folder mean_output/")
