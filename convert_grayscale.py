from PIL import Image

img1 = Image.open("landscape_citra.png")
img2 = Image.open("potrait_citra.png")

img1_gray = img1.convert("L")   # "L" artinya grayscale
img2_gray = img2.convert("L")

img1_gray.save("hasil_landscape_grayscale.png")
img2_gray.save("hasil_portrait_grayscale.png")

print("Konversi berhasil! File grayscale tersimpan.")
