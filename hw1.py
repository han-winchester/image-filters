import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



#==========================================================================================================================================================================
# Convolution
# Takes in the image and a kernel
# This method applies padding to the image then finds the correlation then convolves the kernel with the image
# returns the filtered image
def convolution(img, kernel):

    imgW, imgH = img.shape # get image dimensions
    kernelW, kernelH = kernel.shape # get kernel dimensions
    filtImg = np.ones((imgW, imgH)) # intialize output image

    # pad the image
    padX = (kernelW - 1) // 2 # get the vertical pad length 
    padY = (kernelH - 1) // 2 # get the horizontal pad length
    padImg = np.zeros((imgW + (2 * padX), imgH + (2 * padX))) # fill the border with zeros
    padImgW, padImgH = padImg.shape # get dimensions of the new padded matrix
    padImg[padX:padImgW - padX, padY:padImgH - padY] = img # insert image into the padded matrix

    # iterate through the image; convolve the kernel and image patch by patch
    for i in range(imgW):
        for j in range(imgH):
            # get a patch of the image and multiply with the kernel then sum up all the values to get a pixel value for the filtered image
            filtImg[i, j] = np.sum(kernel * padImg[i:i + kernelW, j:j + kernelH]) 
    
    # returns the filtered image
    return filtImg

#==========================================================================================================================================================================
def filter1D(img, kernel):
    imgW, imgH = img.shape
    kernelW = len(kernel)
    kradius = (kernelW - 1) // 2

    filtImg = np.zeros(img.shape[0])
    
    for i in range(kradius, img.shape[0]-kradius):
        patch = img[i-kradius:i+kradius+1]
        filtImg[i] = np.sum((patch * kernel[::-1]))
    return filtImg

#==========================================================================================================================================================================
# Q1 Box Filter
# Function to apply the box filter on an image, replaces each pixel with the average of its neighborhood. The neighborhood is dependent on the size of the kernel
def boxFilter(img, kernelSize):
    #cv.imshow("Not Filtered", img)
    kernel = np.zeros((kernelSize,kernelSize))     
    matrix = np.ones((kernelSize,kernelSize), dtype=np.float64) # create and return the nxn matrix filled with ones 
    
    # calculate the kernel average; returns the kernel matrix of the average
    kernel = matrix / (kernelSize ** 2) 
    
    # Convolve kernel with image
    filtImg = convolution(img, kernel)

    # output image
    cv.imwrite("Box Filter k=" + str(kernelSize) +".png", filtImg)# saves image to files; same directory as this program
    
    cv.waitKey(0) # waits for user to close the window(s)

#==========================================================================================================================================================================
# Q2 Median Filter
# Function to apply a median filter to an image, based on the size of the kernel it will get the median of the sorted values in this array and replace the middle pixel with this median
def medianFilter(img, kernelSize):
    #cv.imshow("Not Filtered", img)

    # get dimensions of the image
    imgW, imgH = img.shape 

    kernel = [] # create the 1d kernel array, easy to sort and find the middle number
    filtImg = np.zeros((imgW,imgH)) # initialize output image with a matrix of zeroes

    x = (kernelSize // 2) # horizontal edge
    y = (kernelSize // 2) # vertical edge
    for i in range(x, imgW - x): # iterate through imgHeight of image leaving out the boundaries
        for j in range(y, imgH - y): # iterate through imgWidth of image leaving out the boundaries
            for k in range (kernelSize): # iterate through the nxn kernel matrix
                for l in range (kernelSize): 
                    kernel.append(img[i + k - x, j + l - y])
            kernel.sort() # ascending order
            filtImg[i,j] = kernel[(kernelSize**2)//2] # finds the value at the middle position of the kernel matrix
            kernel = [] # makes a new empty nxn kernel matrix for next iteration

    cv.imwrite("Median Filter k=" + str(kernelSize) +".png", filtImg)# saves image to files; same directory as this program
    
    cv.waitKey(0)

#==========================================================================================================================================================================
# Q3 Gaussian Filter
# Function to implement Gaussian Filter, using the gaussian function to calculate the kernel pixels then convolves the kernel to the image matrix
def gaussianFilter(img, sigma):
    #cv.imshow("Not Filtered", img)

    kernel = np.zeros((sigma,sigma))
    
    # iterate through the kernel window and apply the gaussian function
    for i in range(sigma): 
        for j in range(sigma):
            kernel[i, j] = (1 / ((2 * np.pi * (sigma ** 2))) * np.exp(-(((i ** 2) + (j ** 2))/(2*(sigma** 2))))) # 2D Gaussian Function       

    # Convolve the kernel with the image
    filtImg = convolution(img, kernel)

    # Scale the image
    scalefiltImg = 255*(filtImg/np.max(filtImg))

    # Output the image
    cv.imwrite("Gaussian Filter s=" + str(sigma) +".png", scalefiltImg) # saves image to files; same directory as this program

    cv.waitKey(0)
    
    # returns the blurred image; used in canny edge
    return scalefiltImg
     
#==========================================================================================================================================================================
# Q4 Gradient Operations
def gradientOperations(img):
    #cv.imshow("Not Filtered", img)

    # Backward Difference Convolution
    backDiffMatX = [[-1, 1], [-1, 1]] # backward difference in x direction
    backDiffX = np.asarray(backDiffMatX)
    backDiffXImg = convolution(img, backDiffX) # convole in x direction
    backDiffMatY = [[-1, -1], [1, 1]] # backward difference in y direction
    backDiffY = np.asarray(backDiffMatY)
    backDiffYImg = convolution(img, backDiffY) # convole in y direction

    # calculate magnitude
    backMagFiltImg = np.sqrt((backDiffXImg * backDiffXImg) + (backDiffYImg * backDiffYImg)) 

    # scale the X, Y and Mag images
    backDiffXImg = np.abs(backDiffXImg)
    backDiffYImg = np.abs(backDiffYImg)
    backDiffXImgScale = 255 * ((backDiffXImg - np.min(backDiffXImg))/(np.max(backDiffXImg) - np.min(backDiffXImg)))
    backDiffYImgScale = 255 * ((backDiffYImg - np.min(backDiffYImg))/(np.max(backDiffYImg) - np.min(backDiffYImg)))
    backDiffScaleImg = 255 * ((backMagFiltImg - np.min(backMagFiltImg))/(np.max(backMagFiltImg) - np.min(backMagFiltImg)))

    # output X, Y and Mag images
    cv.imwrite("Gradient Operations Back X.png", backDiffXImgScale) 
    cv.imwrite("Gradient Operations Back Y.png", backDiffYImgScale)
    cv.imwrite("Gradient Operations Back Magnitude Scaled.png", backDiffScaleImg)

    #--------------------------------------------------------------------------------------------------------------
    # Forward Difference Convolution
    forwardDiffMatX = [[1, -1], [1, -1]] # forward difference in x direction
    forwardDiffX = np.asarray(forwardDiffMatX)
    forwardDiffXImg = convolution(img, forwardDiffX) # convole in x direction
    forwardDiffMatY = [[1, 1], [-1, -1]] # forward difference in y direction
    forwardDiffY = np.asarray(forwardDiffMatY)
    forwardDiffYImg = convolution(img, forwardDiffY) # convole in y direction

    # calculate magnitude
    forwardMagFiltImg = np.sqrt(np.square(forwardDiffXImg) + np.square(forwardDiffYImg))
    
    # Scale X, Y, and Mag Images
    forwardDiffXImg = np.abs(forwardDiffXImg)
    forwardDiffYImg = np.abs(forwardDiffYImg)
    forwardXImgScale = 255 * ((forwardDiffXImg - np.min(forwardDiffXImg))/(np.max(forwardDiffXImg) - np.min(forwardDiffXImg)))
    forwardYImgScale = 255 * ((forwardDiffYImg - np.min(forwardDiffYImg))/(np.max(forwardDiffYImg) - np.min(forwardDiffYImg)))
    forwardDiffScaleImg = 255 * ((forwardMagFiltImg - np.min(forwardMagFiltImg))/(np.max(forwardMagFiltImg) - np.min(forwardMagFiltImg)))

    # Output X, Y and Mag images
    cv.imwrite("Gradient Operations Forward X.png", forwardXImgScale) 
    cv.imwrite("Gradient Operations Forward Y.png", forwardYImgScale)
    cv.imwrite("Gradient Operations Forward Magnitude Scaled.png", forwardDiffScaleImg)

    #--------------------------------------------------------------------------------------------------------------
    # Center Difference Convolution
    centerDiffMatX = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] # center differece in x direction
    centerDiffX = np.asarray(centerDiffMatX)
    centerDiffXImg = convolution(img, centerDiffX) # convole in x direction
    centerDiffMatY = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] # center differece in y direction
    centerDiffY = np.asarray(centerDiffMatY)
    centerDiffYImg = convolution(img, centerDiffY) # convole in y direction

    # Calculate Mag
    centerMagFiltImg = np.sqrt(np.square(centerDiffXImg) + np.square(centerDiffYImg)) # calculate magnitude

    # Scale X, Y, and Mag images
    centerDiffXImg = np.abs(centerDiffXImg)
    centerDiffYImg = np.abs(centerDiffYImg)
    centerXImgScale = 255 * ((centerDiffXImg - np.min(centerDiffXImg))/(np.max(centerDiffXImg) - np.min(centerDiffXImg)))
    centerYImgScale = 255 * ((centerDiffYImg - np.min(centerDiffYImg))/(np.max(centerDiffYImg) - np.min(centerDiffYImg)))
    centerDiffScaleImg = 255 * ((centerMagFiltImg - np.min(centerMagFiltImg))/(np.max(centerMagFiltImg) - np.min(centerMagFiltImg)))

    # output X, Y, and Mag images
    cv.imwrite("Gradient Operations Center X.png", centerXImgScale) 
    cv.imwrite("Gradient Operations Center Y.png", centerYImgScale)
    cv.imwrite("Gradient Operations Center Magnitude Scaled.png", centerDiffScaleImg)

    cv.waitKey(0)

    # returns the gradient orientation of the image; used in canny edge detector
    gradOrientation = np.degrees(np.arctan2(centerDiffYImg,centerDiffXImg)) 
    return (centerDiffScaleImg, gradOrientation)

#==========================================================================================================================================================================
# Q5 Sobel Filtering
def sobelFilter(img, kernelSize):
    #cv.imwrite("Not Filtered.png", img)
    #cv.imshow("Not Filtered", img)

    # initialize sobel matrix to determine vertical edges
    matrixY = [[1, 0, -1], [2, 0, -2], [1, 0, -1]] 
    kernelY = np.asarray(matrixY)
    # initialize sobel matrix to determine horizontal edges
    matrixX = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] 
    kernelX = np.asarray(matrixX)
  
    # apply the kernel to image by convolution
    filtImgY = convolution(img, kernelY)
    filtImgX = convolution(img, kernelX)

    filtImg = np.sqrt(np.square(filtImgX) + np.square(filtImgY))
    
    # output the image
    cv.imwrite("Sobel FilterX k=" + str(kernelSize) +".png", filtImgX) # saves image to files; same directory as this program
    cv.imwrite("Sobel FilterY k=" + str(kernelSize) +".png", filtImgY) # saves image to files; same directory as this program
    cv.imwrite("Sobel Filter k=" + str(kernelSize) +".png", filtImg) # saves image to files; same directory as this program

    cv.waitKey(0)

#==========================================================================================================================================================================
# Q6 Fast Gaussian Filter
def fastGaussianFilter(img, sigma):
    #cv.imshow("Not Filtered", img)
    filtImg = np.zeros((img.shape[0]))

    kernelSize = sigma
    kradius = int(np.floor(kernelSize / 2))
    kernel = np.linspace(-kradius, kradius, kernelSize)
   

    for i in range(sigma):
        kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(i/(sigma ** 2))) # 1D Gaussian Function

    kernel = kernel / np.sum(kernel)
    

    filtImgx = filter1D(img, kernel)
    filtImgy = filter1D(img, np.transpose(kernel))
    filtImg = np.sqrt(np.square(filtImgx) + np.square(filtImgy))

    cv.imwrite("Fast Gaussian Filter s=" + str(sigma) +".png", filtImg)
    cv.imshow("Fast Gaussian Filter s=" + str(sigma), filtImg)
    #cv.imwrite("Fast Gaussian FilterX s=" + str(sigma) +".png", filtImgX) # saves image to files; same directory as this program
    #cv.imwrite("Fast Gaussian FilterY s=" + str(sigma) +".png", filtImgY) # saves image to files; same directory as this program
    #cv.imwrite("Function GFilter s=" + str(sigma)+".png", cv.GaussianBlur(img,(0,0),sigma)) # built in function; use to compare with my function

    cv.waitKey(0)

#==========================================================================================================================================================================
# Q7 Histogram
# Calculates and plots the histogram of a given image and number of bins
def histogram(img, bins):
    binCount = np.zeros(bins)
    binRange = 256 // bins # gets the range of bin values
    imgW, imgH = img.shape

    # iterate through image; calculate how many times a pixel value occurs in the range of bins
    for i in range(imgW):
        for j in range(imgH):
            pixelVal = img[i, j] // binRange
            binCount[pixelVal] += 1

    # output the histogram
    plt.figure()
    hist = plt.bar(np.arange(bins), binCount)
    plt.xlabel("The number of Bins; B=" + str(bins) +  "; Bin range=" + str(binRange))
    plt.ylabel("Number of times a pixel within the range occurs")
    plt.show()
    return hist, binCount

#==========================================================================================================================================================================
# Used in Canny Edge Detection
# Takes in the img after applying non max suppression
# returns the image after checking each medium pixel for strong pixels in its neighborhood
def hysteresis(img):
    # initialize high and low thresholds
    highThreshold = 90
    lowThreshold = 20

    # Apply Hysteresis
    medium = np.zeros((img.shape)) # medium array for pixels between high and low threshold
    mediumW, mediumH = medium.shape

    # iterate through the filtered image and find strong and weak pixels
    for i in range(1,img.shape[0]):
        for j in range(1,img.shape[1]):
            if(img[i, j] < lowThreshold):
                img[i, j] = 0 # weak pixel
            elif(img[i, j] > highThreshold):
                img[i, j] = 1 # strong pixel
            else:
                np.append(medium, img[i, j]) # if between the high and low threshold

    isBetween = True
    while(isBetween):
        isBetween = False
        # iterate through the medium pixels array to find if pixels are connected to strong pixels
        for i in range(mediumW):
            for j in range(mediumH):
                if(medium[i, j] > 0):
                    patch = img[i:i+mediumW, j:j+mediumH]
                    if(np.sum(patch) > 0):
                        img[i, j] = 1
                        medium[i, j] = 0
                        isBetween = True
    return img

#==========================================================================================================================================================================
# Used in Canny Edge Detection
# Takes in the blurred image and it's gradient orientation. Then determines based on the angle of the orientation it gets the pixel values in that direction and compares with current image to determine if it's an edge
# Returns the image after applying non-max suppression
def nonMaxSuppression(img, gradOrient):
    imgWidth, imgHeight = img.shape
    suppressedImg = np.zeros((img.shape))
    p, r = 0, 0

    # Apply non-max suppression
    for i in range(1, imgWidth - 1): # iterate through the image 
        for j in range(1, imgHeight - 1):
            # horizontal; if the gradient orientation falls within this range assign the pixel values left and right of the image pixel
            if ((0 <= gradOrient[i, j] <= 22.5) or (157.5 <= gradOrient[i,j] <= 180) or (-22.5 <= gradOrient[i, j] <= 0) or (-180 <= gradOrient[i,j] <= -157.5)):
                p = img[i, j+1]
                r = img[i, j-1]
            # diagonal 45; if the gradient orientation falls within this range assign the pixel values diagonal at a 45 degree angle of the image pixel
            elif ((22.5 <= gradOrient[i,j] < 67.5) or (-67.5 < gradOrient[i,j] <= -22.5)):
                p = img[i+1, j-1]
                r = img[i-1, j+1]
            # vertical; if the gradient orientation falls within this range assign the pixel values below and above the image pixel
            elif (67.5 <= gradOrient[i,j] < 112.5) or (-112.5 < gradOrient[i,j] <=-67.5):
                p = img[i+1, j]
                r = img[i-1, j]
            # diagonal 135; if the gradient orientation falls within this range assign the pixel values diagonal at a 135 degree angle of the image pixel
            elif (112.5 <= gradOrient[i,j] < 157.5) or (-157.5 < gradOrient[i,j] <= -112.5):
                p = img[i-1, j-1]
                r = img[i+1, j+1]

            # find the edge
            if (img[i,j] >= p) and (img[i,j] >= r):
                suppressedImg[i,j] = img[i,j] # if it is an edge
            else:
                suppressedImg[i,j] = 0 # not an edge
    
    return suppressedImg

#==========================================================================================================================================================================
# Q8 Canny Edge Detection
def cannyEdge(img, sigma):
    imgWidth, imgHeight = img.shape
    imgBlur = gaussianFilter(img, sigma)  # apply gaussian filter to blur image
    imgBlurGrad, gradOrient = gradientOperations(imgBlur) # apply gradient operations and get the gradient orientation
    #cv.imwrite("Canny Edge Canny1 s=" + str(sigma) + ".png", imgBlurGrad)
    
    # Apply non max suppression to blurred image
    filtImg = nonMaxSuppression(imgBlurGrad, gradOrient)
    
    # output the image after non-max suppression
    cv.imwrite("CE Suppressed s=" + str(sigma) +".png", filtImg)

    # Apply hysteresis to non max suppressed image
    filtImg = hysteresis(filtImg)

    scalefiltImg = 255*(filtImg/np.max(filtImg))

    # output the final canny edge image
    cv.imwrite("CE Final Scale s=" + str(sigma) +".png", scalefiltImg)
    cv.imwrite("CE Final s=" + str(sigma) +".png", filtImg)
    cv.imshow("CE Final s=" + str(sigma), filtImg)
    cv.waitKey(0)

#==========================================================================================================================================================================
# Q9 Image Segmentation
# converts the image to an image of 1 and 0
def binarization(img, threshold, flag):
    imgWidth, imgHeight = img.shape
    filtImg = np.zeros((imgWidth,imgHeight)) # initialize the filtered img

    # binarization 
    for i in range(imgWidth): # iterate through the img
        for j in range(imgHeight):
            if img[i, j] > threshold:
                filtImg[i, j] = 1 # if the pixel value is greater than the threshold replace with 0
            else:
                filtImg[i, j] = 0 # else replace with 1

    # output image
    cv.imwrite("img seg pic=" + str(flag) + " t=" + str(threshold) + ".png", filtImg)
    cv.imshow("img seg pic=" + str(flag) + " t=" + str(threshold), filtImg)

    cv.waitKey(0)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Otsu
# Algorithm to determine the threshold of an image automatically. Uses the histogram of the image to calculate the sum and mean to calculate the variance.
def otsu(img, flag):

    s, P1, P2, mu1, mu2, threshold = 0, 0, 0, 0, 0, 0
  
    histogram(img, 256)
    hist, bins = np.histogram(img, 256) # returns histogram as an array
 
    # while not at the end of the histogram calculate the sum and mean from both sides of i from the graph then calculate the variance
    for i in range(1, 256):
        # calculate the sum
        P1 = np.sum(hist[:i]) # sum the values of the histogram to the left of col i
        P2 = np.sum(hist[i:]) # sum the values of the histogram to the right of col i

        # Calculate the mean
        mu1 = np.mean(hist[:i]) # get mean of values left of col i
        mu2 = np.mean(hist[i:]) # get mean of values right of col i

        # Calculate the variance
        variance = P1 * P2 * ((mu1-mu2)**2)

        # if variance is greater than s then get new threshold
        if(variance > s):
            s = variance
            threshold =  i

    # after it calculates the threshold apply binarization with it
    binarization(img, threshold, flag)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Image Segmentation
# This function only applies binarization at various thresholds. The flag is just to help differentiate between the images when outputting
def imageSeg(img, flag):
    #cv.imshow("Not Filtered", img)
    pixelCount = histogram(img, 256) # plot the histogram of the img to determine where to get the threshold
   
   # apply binarization with manual thresholds
    if(flag == 1): # if image 1
        binarization(img, 130, flag) # threshold = 130
        binarization(img, 170, flag) # threshold = 170
        binarization(img, 65, flag) # threshold = 65

    elif(flag == 2): # if image 2
        binarization(img, 80, flag)
        binarization(img, 60, flag)
        binarization(img, 50, flag)

    elif(flag == 3): # if image 3
        binarization(img, 100, flag)
        binarization(img, 140, flag)
        binarization(img, 60, flag)

#==========================================================================================================================================================================
# Main

# read in the images as greyscale
image1 = cv.imread('images\image1.png', 0) 
image2 = cv.imread('images\image2.png', 0)
image3 = cv.imread('images\image3.png', 0)
image4 = cv.imread('images\image4.png', 0)
picture1 = cv.imread('images\picture1.png', 0)
picture2 = cv.imread('images\cowboy.jpg', 0) 
picture3 = cv.imread('images\king.jpg', 0) 
canny1 = cv.imread('images\canny1.jpg', 0)
canny2 = cv.imread('images\canny2.jpg', 0)


# note: Uncomment whatever functions you want to call. If there are functions with more than 1 image then
# it will be best to uncomment only the functions to work on one image at a time. 
# Example; If you uncomment boxFilter(image1) and boxfilter(image2) then the output for image 2 may overwrite the output for image 1. 



# Q1 Box Filter
#boxFilter(image1, 3) # kernal size of 3
#boxFilter(image1, 5) # kernal size of 5
#boxFilter(image2, 3) # kernal size of 3
#boxFilter(image2, 5) # kernal size of 5
    
# Q2 Median Filter
#medianFilter(image1, 3) # kernal size of 3
#medianFilter(image1, 5) # kernal size of 5
#medianFilter(image1, 7) # kernal size of 7
#medianFilter(image1, 11) # kernal size of 7
#medianFilter(image2, 3) # kernal size of 3
#medianFilter(image2, 5) # kernal size of 5
#medianFilter(image2, 7) # kernal size of 7
#medianFilter(image2, 11) # kernal size of 7

# Q3 Gaussian Filter
#gaussianFilter(image1, 3)  # sigma of 3
#gaussianFilter(image1, 5)  # sigma of 5
#gaussianFilter(image1, 11) # sigma of 10
#gaussianFilter(image2, 3)  # sigma of 3
#gaussianFilter(image2, 5)  # sigma of 5
#gaussianFilter(image2, 11) # sigma of 10

# Q4 Gradient Operations
#gradientOperations(image3)

# Q5 Sobel Filter
#sobelFilter(image1, 3) # kernel size of 3
#sobelFilter(image2, 3) # kernel size of 3

# Q6 Fast Gaussian Filter
#fastGaussianFilter(image1, 3)  # sigma of 3
#fastGaussianFilter(image1, 5)  # sigma of 5
#fastGaussianFilter(image1, 11) # sigma of 10
#fastGaussianFilter(image2, 3)  # sigma of 3
#fastGaussianFilter(image2, 5)  # sigma of 5
#fastGaussianFilter(image2, 10) # sigma of 10

# Q7 Histogram
#histogram(image4, 256) # 256 bins
#histogram(image4, 128) # 128 bins
#histogram(image4, 64)  # 64 bins

# Q8 Canny Edge Detection
#cannyEdge(canny1, 1) # sigma of 1
#cannyEdge(canny1, 3) # sigma of 3
#cannyEdge(canny1, 5) # sigma of 5
#cannyEdge(canny2, 1) # sigma of 1
#cannyEdge(canny2, 3) # sigma of 3
#cannyEdge(canny2, 5) # sigma of 5

# Q9 Image Segmentation
#Binarization with manual threshold
#imageSeg(picture1, 1)
#imageSeg(picture2, 2)
#imageSeg(picture3, 3)

# Binarization with Otsu Threshold
#otsu(image4, 1)
#otsu(picture2, 2)
#otsu(picture3, 3)