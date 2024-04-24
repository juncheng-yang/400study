#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import argparse
import os
import math

# Function to resize the image for display
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

# Function to apply a mask to the image based on the region of interest
def mask_region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to process the street image and optimize lane detection
def process_street_image_optimized(image_path, clip_limit, gauss_kernel_size, canny_threshold1, canny_threshold2, save_steps=False):
    # Read the image
    image = cv2.imread(image_path)

    # Resize image for easier handling, if needed
    image = resize_image_for_display(image)
    if save_steps:
        cv2.imwrite(os.path.splitext(image_path)[0] + '_resized.jpg', image)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save_steps:
        cv2.imwrite(os.path.splitext(image_path)[0] + '_gray.jpg', gray)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (gauss_kernel_size, gauss_kernel_size), 0)
    if save_steps:
        cv2.imwrite(os.path.splitext(image_path)[0] + '_blurred.jpg', blurred)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    if save_steps:
        cv2.imwrite(os.path.splitext(image_path)[0] + '_edges.jpg', edges)

    # Define a trapezoidal region of interest to focus on the road
    height, width = image.shape[:2]
    roi_vertices = np.array([[
        (width * 0.1, height),  # Bottom left
        (width * 0.45, height * 0.6),  # Top left
        (width * 0.55, height * 0.6),  # Top right
        (width * 0.9, height)  # Bottom right
    ]], dtype=np.int32)

    # Apply mask to the region of interest on the image
    roi_image = mask_region_of_interest(edges, roi_vertices)
    if save_steps:
        cv2.imwrite(os.path.splitext(image_path)[0] + '_roi.jpg', roi_image)

    # Hough lines detection
    lines = cv2.HoughLinesP(roi_image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=15, maxLineGap=20)
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) else float('inf')  # Avoid division by zero
                if math.fabs(slope) < 0.5:  # Ignore nearly horizontal lines
                    continue
                if slope <= 0:  # Left line
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:  # Right line
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
        if save_steps:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.imwrite(os.path.splitext(image_path)[0] + '_hough.jpg', line_image)

    # Draw the lines on the image
    line_image = draw_lines(image, lines)
    if save_steps:
        cv2.imwrite(os.path.splitext(image_path)[0] + '_lines_on_image.jpg', line_image)

    # Save the final result image
    result_path = os.path.splitext(image_path)[0] + '_optimized.jpg'
    cv2.imwrite(result_path, line_image)

    return result_path

# Function to draw the lines on the image
def draw_lines(image, lines):
    if lines is None:
        return image
    # Create a copy of the original image to draw lines on
    line_image = np.copy(image)
    # Define colors for left and right lines
    color_left = [255, 0, 0]  # Red color for left line
    color_right = [0, 255, 0]  # Green color for right line
    # Separate lines into left and right based on the slope
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) else float('inf')  # Avoid division by zero
            if math.fabs(slope) < 0.5:  # Ignore nearly horizontal lines
                continue
            if slope <= 0:  # Left line
                left_lines.append(line)
            else:  # Right line
                right_lines.append(line)
    # Draw left lines
    for line in left_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color_left, 10)
    # Draw right lines
    for line in right_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color_right, 10)
    return line_image

# Argument parser setup
parser = argparse.ArgumentParser(description='Process a street image to detect lanes.')
parser.add_argument('image_path', type=str, help='The path to the image file.')
parser.add_argument('--clip_limit', type=float, default=2.0, help='CLAHE clip limit.')
parser.add_argument('--gauss_kernel_size', type=int, default=9, help='Gaussian blur kernel size.')
parser.add_argument('--canny_threshold1', type=int, default=50, help='Canny edge detection lower threshold.')
parser.add_argument('--canny_threshold2', type=int, default=150, help='Canny edge detection upper threshold.')
parser.add_argument('--save_steps', action='store_true', help='Save images at each processing step.')

if __name__ == '__main__':
    args = parser.parse_args()
    # Assuming the image is in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(script_dir, args.image_path)

    processed_image_path = process_street_image_optimized(
        full_image_path,
        args.clip_limit,
        args.gauss_kernel_size,
        args.canny_threshold1,
        args.canny_threshold2,
        args.save_steps
    )
    print(f"Processed image saved at: {processed_image_path}")


# In[ ]:




