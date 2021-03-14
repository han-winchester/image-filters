import numpy as np
import cv2
from scipy import ndimage
from scipy import linalg
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import pdb 
    

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


def gaussian_kernel(size=3, sigma=1):
    '''
    Creates Gaussian Kernel (1 Dimensional)
    Inputs: 
      size: width of the filter
      sigma: standard deviation
    Returns a 1xN shape 1D gaussian kernel
    '''
    
    half_size = int(size) // 2
    x = np.mgrid[-half_size:half_size + 1]
    mu = 0
    
    # Apply gaussian formula
    #g = np.exp(-((x-np.mean(x)) ** 2 / (2.0 * sigma ** 2))) / (np.sqrt(2.0 * np.pi) * sigma ** 2)
    g = np.exp(-((x-mu) ** 2 / (2.0 * sigma ** 2))) / (np.sqrt(2.0 * np.pi) * sigma ** 2)
    
    # Normalize values 
    g = g / np.sum(g)
    
    g = g.reshape((1, g.shape[0]))
    #print(g)
    return g
    
    
def gaussian_first_derivative_kernel(size=3, sigma=1):
    '''
    Creates 1st derviative gaussian Kernel (1 Dimensional)
    Inputs: 
      size: width of the filter
      sigma: standard deviation
    Returns a 1xN shape 1D 1st derivative gaussian kernel
    '''
    
    half_size = int(size) // 2
    x = np.mgrid[-half_size:half_size + 1]
    mu = 0
    g = (-1) * ((x - mu)/ (sigma ** 2)) * np.exp(-((x - mu) ** 2 / (2.0 * sigma ** 2))) / (np.sqrt(2.0 * np.pi) * sigma ** 2)
    #g = (-1) * ((x - np.mean(x))/ (sigma ** 2)) * np.exp(-((x - np.mean(x)) ** 2 / (2.0 * sigma ** 2))) / (np.sqrt(2.0 * np.pi) * sigma ** 2)
    g = g / np.sum(g)
    g = g.reshape((1, g.shape[0]))
    #print(g)
    return g
    

def non_max_supression(det, phase):
    '''
    Performs non-maxima supression for given magnitude and orientation.
    Returns output with nms applied. Also return a colored image based on gradient direction for maximum value.
    '''
    
    gmax = np.zeros(det.shape)
    color_o = np.zeros((det.shape[0], det.shape[1], 3))
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
          if phase[i][j] < 0:
            phase[i][j] += 360

          if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
            # 0 degrees
            if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
              if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                gmax[i][j] = det[i][j]
              color_o[i][j] = [0, 0, 255]
            # 45 degrees
            if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
              if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
                gmax[i][j] = det[i][j]
              color_o[i][j] = [0, 255, 255]
            # 90 degrees
            if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
              if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
                gmax[i][j] = det[i][j]
              color_o[i][j] = [255, 0, 255]
            # 135 degrees
            if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
              if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
                gmax[i][j] = det[i][j]
              color_o[i][j] = [0, 255, 0]
    return gmax, color_o


def DFS(img):
    '''
    If pixel is linked to a strong pixel in a local window, make it strong as well.
    Called iteratively to make all strong-linked pixels strong.
    '''
    
    for i in range(1, int(img.shape[0] - 1)) :
        for j in range(1, int(img.shape[1] - 1)) :
            if(img[i, j] == 1) :
                t_max = max(img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1],
                            img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j+1])
                if(t_max == 2) :
                    img[i, j] = 2
                
                    
def hysteresis_thresholding(img, low_ratio, high_ratio):
    diff = np.max(img) - np.min(img)
    t_low = np.min(img) + low_ratio * diff
    t_high = np.min(img) + high_ratio * diff
    
    temp_img = np.copy(img)
    
    #Assign values to pixels
    for i in range(1, int(img.shape[0] - 1)) :
        for j in range(1, int(img.shape[1] - 1)) :
            #Strong pixels
            if(img[i, j] > t_high) :
                temp_img[i, j] = 2
            #Weak pixels
            elif(img[i, j] < t_low) :
                temp_img[i, j] = 0
            #Intermediate pixels
            else :
                temp_img[i, j] = 1
    
    #Include weak pixels that are connected to chain of strong pixels 
    total_strong = np.sum(temp_img == 2)
    while(1) :
        DFS(temp_img)
        if(total_strong == np.sum(temp_img == 2)) :
            break
        total_strong = np.sum(temp_img == 2)
    
    #Remove weak pixels
    for i in range(1, int(temp_img.shape[0] - 1)) :
        for j in range(1, int(temp_img.shape[1] - 1)) :
            if(temp_img[i, j] == 1) :
                temp_img[i, j] = 0
    
    temp_img = temp_img/np.max(temp_img)
    return temp_img
    
    
if __name__ == '__main__':
    # Initialize values 
    sigma = 3
    size = 7 # 6 * sigma + 1        # Rule of thumb is 6sigma+1, but you can use fixed kernel size to run it faster
    high_th = 0.3       # Between 0 to 1 here, so its 255 * 0.3 = 76.5
    low_th = 0.1
    image_name = 'canny1.jpg'
    
    # Read the image in grayscale mode using opencv
    I = cv2.imread(image_name,0)

    I = I.astype(np.float)
    
    # Create a gaussian kernel 1XN matrix
    G = gaussian_kernel(size,sigma)
    
    # Get the First Derivative Kernel
    G_prime = gaussian_first_derivative_kernel(size,sigma)

    der_ker = np.array([[-1,0,1]])  # Derivative kernel (central). You can also use Sobel

    '''
    If you get derivative of Gaussian directly as kernel, you only need to use one kernel.
    This is because these two operations can be combined.
    
    If you use other derivative kernel (Sobel, or central [-1,0,1]), then you have to first
    smoothen (blur) the image. So apply the Gaussian kernel first and then apply the derivative kernel on smooth image.    
    '''
    # Convolution of G and I
    # I_x = convolution_image(I, G)
    # I_y = convolution_image(I, G.T)
    # cv2.imshow("1_Gaus_x_"+image_name,I_x.astype(np.uint8))
    # cv2.imshow("2_Gaus_y_"+image_name,I_y.astype(np.uint8))
    # I_xx = convolution_image(I_x, der_ker)
    # I_yy = convolution_image(I_y, der_ker.T)


    # Derivative of Gaussian Convolution
    I_xx = convolution_image(I, G_prime)
    I_yy = convolution_image(I, G_prime.T)
    
    # Convert derivative result to 0-255 for display.
    # Need to scale from 0-1 to 0-255.
    abs_grad_x = (( (I_xx - np.min(I_xx)) / (np.max(I_xx) - np.min(I_xx)) ) * 255.).astype(np.uint8)
    abs_grad_y = (( (I_yy - np.min(I_yy)) / (np.max(I_yy) - np.min(I_yy)) ) * 255.).astype(np.uint8)
    cv2.imshow("3_Der_x_"+image_name, abs_grad_x)
    cv2.imshow("4_Der_y_"+image_name, abs_grad_y)

    # Compute magnitude
    I_mag = np.hypot(I_xx, I_yy)
    I_mag = (I_mag - np.min(I_mag)) / (np.max(I_mag) - np.min(I_mag))     # Normalize between 0-1
    abs_mag = (I_mag * 255.).astype(np.uint8)
    cv2.imshow("5_Mag_"+image_name, abs_mag)

    # Compute orientation
    gradient_orientation = np.degrees(np.arctan2(I_yy, I_xx) )
    
    # Compute non-max suppression
    # I_orient_color is not important for your assignment
    # This just allows you to view colored orientation for each direction
    I_nms, I_orient_color = non_max_supression(I_mag, gradient_orientation)
    I_nms_img = (I_nms * 255.).astype(np.uint8)
    I_orient_color[I_mag<0.2,:] = 0
    cv2.imshow("6_NMS_"+image_name, I_nms_img)
    cv2.imshow("7_Orientation_"+image_name, I_orient_color)
        
    #Compute thresholding and then hysteresis
    I_hys = hysteresis_thresholding(I_nms, low_th, high_th)
    I_hys = (I_hys * 255.).astype(np.uint8)
    cv2.imshow("8_Hysterisis_"+image_name,I_hys)
    cv2.waitKey(0)























