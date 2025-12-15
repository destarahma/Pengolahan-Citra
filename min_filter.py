import cv2
import os
import numpy as np

# Folder input dan output
input_folder = "noise_output"
output_folder = "min_output"

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Daftar semua file dalam folder noise_output
files = os.listdir(input_folder)

# Kernel MIN (bisa diubah ukurannya, contoh 3x3)
kernel_size = 3

print("Memulai proses filter MIN...")

for filename in files:
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Gagal membaca: {filename}")
            continue

        # Filter MIN = erosian
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        min_filtered = cv2.erode(img, kernel)

        # Simpan hasil
        save_path = os.path.join(output_folder, f"MIN_{filename}")
        cv2.imwrite(save_path, min_filtered)

        print(f"Berhasil: {filename} → MIN_{filename}")

print("Selesai! Semua gambar telah difilter dengan MIN filter.")
