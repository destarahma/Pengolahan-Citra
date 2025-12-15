import cv2
import os
import numpy as np

# Folder input dan output
input_folder = "noise_output"
output_folder = "max_output"

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Daftar semua file dalam folder noise_output
files = os.listdir(input_folder)

# Kernel MAX (3x3)
kernel_size = 3

print("Memulai proses filter MAX...")

for filename in files:
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Gagal membaca: {filename}")
            continue

        # Filter MAX = dilasi (ambil nilai maksimum)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        max_filtered = cv2.dilate(img, kernel)

        # Simpan hasil
        save_path = os.path.join(output_folder, f"MAX_{filename}")
        cv2.imwrite(save_path, max_filtered)

        print(f"Berhasil: {filename} → MAX_{filename}")

print("Selesai! Semua gambar telah difilter dengan MAX filter.")
