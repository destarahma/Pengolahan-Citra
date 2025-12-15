Folder `segmentasi` berisi skrip Python untuk perbandingan MSE dan deteksi tepi.

Cara menjalankan:

1. Buat virtual environment (opsional):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Pasang dependensi:

```powershell
pip install -r requirements.txt
```

3. Jalankan skrip, contoh:

```powershell
python hitung_mse_grafik.py
```

Untuk mengunggah ke GitHub: tambahkan remote dan push:

```powershell
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```
