import numpy as np
from PIL import Image
import random
import os

def load_image(path):
    """Load image as numpy array."""
    img = Image.open(path)
    return np.array(img)

def save_image(arr, path):
    """Save numpy array as image."""
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)

# ==========================================================
#                     SALT & PEPPER NOISE
# ==========================================================
def salt_pepper_noise(img, prob):
    noisy = img.copy()
    row, col = img.shape[0], img.shape[1]

    if len(img.shape) == 2:
        # grayscale 2D
        for i in range(row):
            for j in range(col):
                r = random.random()
                if r < prob:
                    noisy[i][j] = 0
                elif r > 1 - prob:
                    noisy[i][j] = 255
    else:
        # RGB 3D
        for i in range(row):
            for j in range(col):
                r = random.random()
                if r < prob:
                    noisy[i][j] = [0, 0, 0]
                elif r > 1 - prob:
                    noisy[i][j] = [255, 255, 255]
    return noisy

# ==========================================================
#                     GAUSSIAN NOISE
# ==========================================================
def gaussian_noise(img, mean, sigma):
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy

# ==========================================================
#                     MAIN PROGRAM
# ==========================================================
if __name__ == "__main__":
    if not os.path.exists("noise_output"):
        os.makedirs("noise_output")

    print("Loading images...")
    img_rgb_land = load_image("landscape_citra.png")
    img_rgb_port = load_image("portrait_citra.png")

    img_gray_land = load_image("hasil_landscape_grayscale.png")
    img_gray_port = load_image("hasil_portrait_grayscale.png")

    prob_low = 0.02
    prob_high = 0.10

    sigma_low = 10
    sigma_high = 30

    # ============== SALT & PEPPER ===================
    print("Applying Salt & Pepper noise...")

    # RGB
    save_image(salt_pepper_noise(img_rgb_land, prob_low), "noise_output/rgb_land_sp_low.png")
    save_image(salt_pepper_noise(img_rgb_land, prob_high), "noise_output/rgb_land_sp_high.png")

    save_image(salt_pepper_noise(img_rgb_port, prob_low), "noise_output/rgb_port_sp_low.png")
    save_image(salt_pepper_noise(img_rgb_port, prob_high), "noise_output/rgb_port_sp_high.png")

    # Grayscale
    save_image(salt_pepper_noise(img_gray_land, prob_low), "noise_output/gray_land_sp_low.png")
    save_image(salt_pepper_noise(img_gray_land, prob_high), "noise_output/gray_land_sp_high.png")

    save_image(salt_pepper_noise(img_gray_port, prob_low), "noise_output/gray_port_sp_low.png")
    save_image(salt_pepper_noise(img_gray_port, prob_high), "noise_output/gray_port_sp_high.png")

    # ============== GAUSSIAN ========================
    print("Applying Gaussian noise...")

    # RGB
    save_image(gaussian_noise(img_rgb_land, 0, sigma_low), "noise_output/rgb_land_gauss_low.png")
    save_image(gaussian_noise(img_rgb_land, 0, sigma_high), "noise_output/rgb_land_gauss_high.png")

    save_image(gaussian_noise(img_rgb_port, 0, sigma_low), "noise_output/rgb_port_gauss_low.png")
    save_image(gaussian_noise(img_rgb_port, 0, sigma_high), "noise_output/rgb_port_gauss_high.png")

    # Grayscale
    save_image(gaussian_noise(img_gray_land, 0, sigma_low), "noise_output/gray_land_gauss_low.png")
    save_image(gaussian_noise(img_gray_land, 0, sigma_high), "noise_output/gray_land_gauss_high.png")

    save_image(gaussian_noise(img_gray_port, 0, sigma_low), "noise_output/gray_port_gauss_low.png")
    save_image(gaussian_noise(img_gray_port, 0, sigma_high), "noise_output/gray_port_gauss_high.png")

    print("=== Noise process completed! Results saved in 'noise_output' folder ===")
