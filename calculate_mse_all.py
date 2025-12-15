import cv2
import numpy as np
import os
from difflib import SequenceMatcher

# folder
original_folder = "original"
filter_folders = ["noise_output", "min_output", "max_output", "median_output", "mean_output"]

report_file = "mse_report_auto.txt"
report_lines = []

def mse(a, b):
    return np.mean((a.astype("float") - b.astype("float")) ** 2)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

print("🔍 Mencari pasangan gambar & menghitung MSE...\n")

# baca semua file original
original_files = [
    f for f in os.listdir(original_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# proses setiap folder filter
for folder in filter_folders:
    report_lines.append(f"\n=== MSE untuk folder: {folder} ===\n")
    print(f"\n=== Folder: {folder} ===")

    filter_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filtered in filter_files:

        # cari file original paling mirip namanya
        best_match = None
        best_score = 0

        for orig in original_files:
            score = similarity(filtered, orig)
            if score > best_score:
                best_score = score
                best_match = orig

        if best_match is None:
            print(f"[SKIP] Tidak ada pasangan untuk {filtered}")
            continue

        # baca gambar
        img_f = cv2.imread(os.path.join(folder, filtered))
        img_o = cv2.imread(os.path.join(original_folder, best_match))

        if img_f is None or img_o is None:
            print(f"[ERROR] Gagal membaca gambar: {filtered}")
            continue

        # samakan ukuran jika berbeda
        if img_f.shape != img_o.shape:
            img_f = cv2.resize(img_f, (img_o.shape[1], img_o.shape[0]))

        # hitung MSE
        mse_value = mse(img_o, img_f)

        info = f"{filtered} -> {best_match} : MSE = {mse_value}"
        print(info)
        report_lines.append(info)

# simpan laporan
with open(report_file, "w") as f:
    for line in report_lines:
        f.write(line + "\n")

print("\nSelesai! Laporan tersimpan di:", report_file)
