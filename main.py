import copy

import numpy as np
import cv2 as cv

img = cv.imread('golbasiCampus-Degraded.jpg')

#img_noise_removed = cv.medianBlur(img,3)
img_noise_removed = cv.GaussianBlur(img, (3  , 3), 0)
img_noise_removed = cv.fastNlMeansDenoisingColored(img_noise_removed,None,3,3,7,21)   # gaussian noise
#img_noise_removed = cv.bilateralFilter(img,10,60,60)

cv.imwrite("noise_removal.jpg", img_noise_removed)

# Contrast enhancement
img_hsv = cv.cvtColor(img_noise_removed, cv.COLOR_RGB2HSV)

# Histogram equalisation on the V-channel
img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])

# convert image back from HSV to RGB
image = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)

cv.imwrite("contrast_Enhanced.jpg", image)

def sharpen():
    x = 'contrast_Enhanced.jpg'
    img = cv.imread(x, 1)

    # ---Sharpening filter----
    kernel = np.array([[0, -1, 0], [0, 4, -1], [0, -1, 0]])
    im = cv.filter2D(img, -1, kernel)
    cv.imwrite("Sharpened.jpg", im)


sharpen()

def binary_image():
    image = cv.imread('Sharpened.jpg')

    gray_scale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # for threshold get  medium value for image
    sum = 0
    for index1 in gray_scale:
        for index2 in index1:
            sum += index2

    height, width = gray_scale.shape
    totalPixels = height * width  # n
    threshold = sum / totalPixels

    im_bw = cv.threshold(gray_scale, threshold, 255, cv.THRESH_BINARY)[1]
    cv.imwrite("new_binary.jpg", im_bw)

    return im_bw


bin_image = binary_image()

def morphological(binary_image):

    kernel = np.ones((3, 3), np.uint8)

    img_erosion = cv.erode(binary_image, kernel, iterations=1)
    img_dilation = cv.dilate(binary_image, kernel, iterations=1)

    opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)

    cv.imshow('Erosion', img_erosion)
    cv.imshow('Dilation', img_dilation)
    cv.imshow('Opening',opening)
    cv.imshow('Closing',closing)

    cv.waitKey(0)

morphological(bin_image)