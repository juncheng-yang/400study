#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import argparse

def resize_image_for_display(image, screen_width_inches=9.8):
    
    screen_width_pixels = int(screen_width_inches * 157) 
    if image.shape[1] > screen_width_pixels:
        scale_percent = screen_width_pixels / image.shape[1]
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized
    else:
        return image

def process_street_image_optimized(image_path, clip_limit, gauss_kernel_size, canny_threshold1, canny_threshold2):
    image = cv2.imread(image_path)
    
    image = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Resize for display
    resized_gray = resize_image_for_display(gray)
    resized_equalized = resize_image_for_display(equalized)
    
    # Show the original grayscale and equalized images for comparison
    cv2.imshow('Original Grayscale Image', gray)
    cv2.imshow('CLAHE Image', equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply Gaussian blur to smooth out noise
    blurred = cv2.GaussianBlur(equalized, (gauss_kernel_size, gauss_kernel_size), 0)

    # Canny edge detection with adjusted thresholds to decrease noise
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Hough lines detection with parameters adjusted to filter out short lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)

    # Drawing the new lines on the original image
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Save the image with lines drawn on it
    result_path = image_path.replace('.jpg', '_optimized.jpg')
    cv2.imwrite(result_path, line_image)

    return result_path


# Parse command line arguments
parser = argparse.ArgumentParser(description='Street image processing with adjustable parameters.')
parser.add_argument('image_path', help='Path to the input image')
parser.add_argument('--clip_limit', type=float, default=2.0, help='Clip limit for CLAHE')
parser.add_argument('--gauss_kernel_size', type=int, default=9, help='Size of the Gaussian kernel')
parser.add_argument('--canny_threshold1', type=int, default=50, help='First threshold for Canny edge detection')
parser.add_argument('--canny_threshold2', type=int, default=150, help='Second threshold for Canny edge detection')
args = parser.parse_args()

# Call the image processing function with the parsed arguments
processed_image_path = process_street_image_optimized(
    args.image_path,
    args.clip_limit,
    args.gauss_kernel_size,
    args.canny_threshold1,
    args.canny_threshold2
)
print(f"Processed image saved at {processed_image_path}")


# In[ ]:




