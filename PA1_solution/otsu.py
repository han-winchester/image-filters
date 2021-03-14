import cv2
import numpy as np
import matplotlib.pyplot as plt

# Three different greyscale images
FILENAMES = ['canny1.jpg', 'canny2.jpg', 'image1.png']

def manual_binarize():
    # Perform tasks on each image
    print("Manual thresholding")
    for i in range(len(FILENAMES)):
        print("Read image: ", FILENAMES[i])

        # Read grayscale image
        img = cv2.imread(FILENAMES[i], 0)
        f_name = FILENAMES[i].split('.')[0]

        bins = 256
        histogram = generate_histogram(img, bins)
        img_thresh = image_binarization(img, threshold=150)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(9, 2)
        fig.suptitle('Manual Threshold')
        ax1.set_title('Image')
        ax1.imshow(img, cmap='gray', aspect='auto')
        ax2.set_title("Histogram")
        ax2.plot(histogram)
        ax3.set_title("Manual threshold")
        ax3.imshow(img_thresh, cmap='gray', aspect='auto')
        plt.savefig(f_name + "_manual.jpg")
        plt.show()


def otsu():
    # Perform tasks on each image
    for i in range(len(FILENAMES)):
        print("Read image: ", FILENAMES[i])

        # Read grayscale image
        img = cv2.imread(FILENAMES[i], 0)

        # Perform Otsu Thresholding
        img_otsu, histogram = otsu_thresholding(img)
        f_name = FILENAMES[i].split('.')[0]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(9,2)
        fig.suptitle('Otsu Threshold')
        ax1.set_title('Image')
        ax1.imshow(img, cmap='gray', aspect='auto')
        ax2.set_title("Histogram")
        ax2.plot(histogram)
        ax3.set_title("Otsu's threshold")
        ax3.imshow(img_otsu, cmap='gray', aspect='auto')
        plt.savefig(f_name + "_otsu.jpg")
        plt.show()

        print("Image saved")



# Perform image binarization
def image_binarization(img, threshold):
    print("Image thresholding")
    img2 = img.copy()
    img2[img2<=threshold] = 0
    img2[img2>threshold] = 255

    return img2


# Perform Otsu Thresholding
def otsu_thresholding(img):
    print("Performing Otsu Thresholding...")

    # Obtain histogram of pixel frequency
    bins = 256
    histogram = generate_histogram(img, bins)
    height, width = img.shape
    total_pixels = height * width

    # Iterate over all possible thresholds
    S = 0
    T = 0
    for u in range(bins):
        # Calculate the variance for the threshold
        variance = calculate_variance(histogram, u, total_pixels)

        # Find the best variance/threshold so far
        if (variance > S):
            S = variance
            T = u

    print("Otsu's Threshold: ", T)

    img_otsu = image_binarization(img, T)

    return img_otsu, histogram

# Creates histogram array of pixel intensities
def generate_histogram(img, bins=256):
    height, width = img.shape
    histogram = np.zeros(bins)
    bin_size = int(256 / bins)      # Range / bins, here range = 256

    # Count pixel intensities for histogram
    for row in range(height):
        for col in range(width):
            intensity = img[row,col]
            intensity_bin = intensity // bin_size
            histogram[intensity_bin] += 1

    return histogram


# Calculates inter-class variance for the histogram at threshold u
def calculate_variance(histogram, u, total_pixels):
    # Count the pixels on each side of u
    pixel_count1 = float(np.sum(histogram[0:u+1]))
    pixel_count2 = float(np.sum(histogram[u+1:]))

    # If no pixel on either side, variance is zero
    if pixel_count1 == 0 or pixel_count2 == 0:
        return 0

    # Sum intensity values on each side of u
    total_pixels = float(total_pixels)
    intensity_sum1 = np.sum(histogram[0:u+1] * np.arange(0, u+1))
    intensity_sum2 = np.sum(histogram[u + 1:] * np.arange(u + 1, 256))

    # Calculate inputs to variance formula
    P1 = pixel_count1 / total_pixels
    P2 = pixel_count2 / total_pixels
    u1 = intensity_sum1 / pixel_count1
    u2 = intensity_sum2 / pixel_count2

    # Calculate the variance
    variance = P1 * P2 * ((u1 - u2) ** 2)

    return variance


if __name__ == '__main__':
    otsu()
    manual_binarize()