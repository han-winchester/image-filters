import cv2
import numpy as np
import matplotlib.pyplot as plt


def convolution_image(image, kernel, average=False, verbose=False):
    '''
    Performs convolution along x and y axis, based on kernel size.
    Assumes input image is 1 channel (Grayscale)
    Inputs:
      image: H x W x C shape numpy array (C=1)
      kernel: K_H x K_W shape numpy array (for example, 3x1 for 1 dimensional filter for y-component)
    Returns:
      H x W x C Image convolved with kernel
    '''

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    kernel = np.flipud(np.fliplr(kernel))

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()
        print("Output Image size : {}".format(output.shape))

    return output


def q1():
    # Read image as grayscale
    img1 = cv2.imread('canny1.jpg', 0)

    kernel = np.ones((5, 5), np.float32) / 25
    manually_filtered = convolution_image(img1, kernel)
    manually_filtered = manually_filtered.astype(np.uint8)

    # Display image
    cv2.imshow("Input Image", img1)
    cv2.imshow("Manually filtered Image", manually_filtered)
    cv2.waitKey(0)

    # Save image to new file
    # cv2.imwrite('q1_img1.jpg', manually_filtered)


def q2():
    # Read image as grayscale
    img1 = cv2.imread('image1.png', 0)

    kernel_size = 5
    manually_filtered = apply_filter_median(img1, kernel_size)
    manually_filtered = manually_filtered.astype(np.uint8)

    # Display image
    cv2.imshow("Input Image", img1)
    cv2.imshow("Median Filter Image", manually_filtered)
    cv2.waitKey(0)

    # Save image to new file
    # cv2.imwrite('q2_img1.jpg', manually_filtered)


def q3():
    # Read image as grayscale
    img1 = cv2.imread('image1.png', 0)

    sigma = 10
    kernel_size = 6*sigma + 1
    kernel_gaus_1d = gaussian_kernel(size=kernel_size, sigma=3)
    kernel_gaus_2d = np.dot(kernel_gaus_1d.T, kernel_gaus_1d)

    manually_filtered = convolution_image(img1, kernel_gaus_2d)
    manually_filtered = manually_filtered.astype(np.uint8)

    # Display image
    cv2.imshow("Input Image", img1)
    cv2.imshow("Manually filtered Image", manually_filtered)
    cv2.waitKey(0)

    # Save image to new file
    # cv2.imwrite('q2_img1.jpg', manually_filtered)


def q4():
    # Read image as grayscale
    img = cv2.imread('image3.png', 0)

    forward_kernel = np.array([[-1,1]])
    backward_kernel = np.array([[1,-1]])
    central_kernel = np.array([[-1,0,1]])

    # cv2.imshow("Input Image", img)

    # --------- Forward kernel -------------
    print("Run forward")
    kernel = forward_kernel
    f_x = convolution_image(img, kernel)
    f_y = convolution_image(img, kernel.T)
    mag = np.hypot(f_x, f_y)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(9,3)
    fig.suptitle('Forward gradient')
    ax1.set_title('F_x')
    ax1.imshow(normalize_image(f_x), cmap='gray')
    ax2.set_title("F_y")
    ax2.imshow(normalize_image(f_y), cmap='gray')
    ax3.set_title("Magnitude")
    ax3.imshow(normalize_image(mag), cmap='gray')
    # plt.savefig("forward_gradient.jpg")
    plt.show()

    # --------- Backward kernel -------------
    print("Run backward")
    kernel = backward_kernel
    f_x = convolution_image(img, kernel)
    f_y = convolution_image(img, kernel.T)
    mag = np.hypot(f_x, f_y)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(9, 3)
    fig.suptitle('Backward gradient')
    ax1.set_title('F_x')
    ax1.imshow(normalize_image(f_x), cmap='gray')
    ax2.set_title("F_y")
    ax2.imshow(normalize_image(f_y), cmap='gray')
    ax3.set_title("Magnitude")
    ax3.imshow(normalize_image(mag), cmap='gray')
    # plt.savefig("Backward_gradient.jpg")
    plt.show()


    # --------- Central kernel -------------
    print("Run central")
    kernel = central_kernel
    f_x = convolution_image(img, kernel)
    f_y = convolution_image(img, kernel.T)
    mag = np.hypot(f_x, f_y)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(9, 3)
    fig.suptitle('Central gradient')
    ax1.set_title('F_x')
    ax1.imshow(normalize_image(f_x), cmap='gray')
    ax2.set_title("F_y")
    ax2.imshow(normalize_image(f_y), cmap='gray')
    ax3.set_title("Magnitude")
    ax3.imshow(normalize_image(mag), cmap='gray')
    # plt.savefig("central_gradient.jpg")
    plt.show()



def q5():
    # Read image as grayscale
    img = cv2.imread('image1.png', 0)

    sobel_x = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    sobel_y = sobel_x.T

    # cv2.imshow("Input Image", img)

    print("Run Sobel")

    f_x = convolution_image(img, sobel_x)
    f_y = convolution_image(img, sobel_y)
    mag = np.hypot(f_x, f_y)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(9,3)
    fig.suptitle('Sobel filtering')
    ax1.set_title('F_x')
    ax1.imshow(normalize_image(f_x), cmap='gray')
    ax2.set_title("F_y")
    ax2.imshow(normalize_image(f_y), cmap='gray')
    ax3.set_title("Magnitude")
    ax3.imshow(normalize_image(mag), cmap='gray')
    # plt.savefig("forward_gradient.jpg")
    plt.show()


def q6():
    # Read image as grayscale
    img = cv2.imread('image1.png', 0)
    img1 = img.astype(float)

    sigma = 5
    kernel_size = 6*sigma + 1
    kernel_gaus_1d = gaussian_kernel(size=kernel_size, sigma=3)

    gaus_x = convolution_image(img1, kernel_gaus_1d)
    gaus_y = convolution_image(img1, kernel_gaus_1d.T)
    gaus = np.hypot(gaus_x, gaus_y)

    # Display image
    cv2.imshow("Input Image", img)
    cv2.imshow("1D Gaus x", normalize_image(gaus_x))
    cv2.imshow("1D Gaus y", normalize_image(gaus_y))
    cv2.imshow("1D Gaus final", normalize_image(gaus))
    cv2.waitKey(0)

    # Save image to new file
    # cv2.imwrite('q6_img1.jpg', gaussian_1d_filtered)


def q7():
    bins = 256
    mat_range = 256
    bin_size = int(mat_range / bins)
    img = cv2.imread("image4.png",0).reshape(-1)

    bin_count = np.zeros((bins))
    for i in range(img.shape[0]):
        this_bin = img[i] // bin_size
        bin_count[this_bin] += 1

    plt.figure(figsize=(4,3))
    plt.plot(bin_count)
    plt.show()

def normalize_image(img):
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.
    norm_img = norm_img.astype(np.uint8)
    return norm_img

def gaussian_kernel(size=3, sigma=1):
    '''
    Creates Gaussian Kernel (1 Dimensional)
    Inputs:
      size: width of the filter
      sigma: standard deviation
    Returns a 1xN shape 1D gaussian kernel
    '''

    half_size = int(size) // 2
    x = np.mgrid[-half_size:half_size + 1]  # This is equivalent to np.linspace(-half_size, half_size, size)
    mu = 0

    # Apply gaussian formula
    # g = np.exp(-((x-np.mean(x)) ** 2 / (2.0 * sigma ** 2))) / (np.sqrt(2.0 * np.pi) * sigma ** 2)
    g = np.exp(-((x - mu) ** 2 / (2.0 * sigma ** 2))) / (np.sqrt(2.0 * np.pi) * sigma ** 2)

    # Normalize values
    g = g / np.sum(g)

    g = g.reshape((1, g.shape[0]))
    # print(g)
    return g

def apply_filter_median(img, kernel_size):
    img_height, img_width = img.shape
    filtered_img = np.zeros((img_height, img_width))
    kernel_radius = int(np.floor(kernel_size / 2.))
    median_index = int((kernel_size*kernel_size)/2.)

    for row in range(kernel_radius, img_height-kernel_radius):
        for col in range(kernel_radius, img_width-kernel_radius):
            # Take image patch, sort it.
            # You can use numpy's sort function for this -> np.sort(patch)
            # Write center value to filtered_img
            patch = img[row-kernel_radius:row+kernel_radius+1,
                        col-kernel_radius:col+kernel_radius+1]
            patch_median = np.sort(patch.reshape(-1))[median_index]
            filtered_img[row, col] = patch_median
    return filtered_img

if __name__ == '__main__':

    # Uncomment to run question number
    q1()
    # q2()
    # q3()
    # q4()
    # q5()
    # q6()
    # q7()
    

