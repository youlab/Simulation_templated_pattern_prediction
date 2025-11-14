"""Describes how the experimental and simulation images were augmented
Note here we just describe how to rotate by a random angle, but in reality we rotate by 100 different angles per image.
Similar to the Seed_DataAugmentation.py file, just example do not run this block here"""

import cv2
import numpy as np
import os
from utils.config import EXPERIMENTAL_FOLDER,SIMULATED_FOLDER


# def rotate_image(img, angle, center=None, scale=1.0):
#     (h, w) = img.shape[:2]
#     if center is None:
#         center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated = cv2.warpAffine(img, M, (w, h))
#     return rotated

def crop_and_rotate_experimental(img, angle):
    """Optimized experimental image rotation using OpenCV warpAffine.
    Steps:
    - detect plate circle with HoughCircles
    - mask and adjust contrast/brightness
    - perform fast rotation with warpAffine (INTER_NEAREST) using negated angle
    - re-apply circular mask and return
    """

    # similar code as some of the preprocessing functions
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in the image using Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles method
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=600)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Take the first circle found
        # Assuming first circle detected by Hough transform is the plate
        x_center, y_center, radius = circles[0]
        new_radius = radius - 82

        # Create a mask where all values are set to zero (black)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (int(x_center), int(y_center)), int(new_radius), 255, -1)
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        
        # Adjust contrast and brightness
        alpha = 1.5
        beta = 50
        adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

        # OPTIMIZED ROTATION using OpenCV warpAffine
        # Use int/float to avoid numpy scalar type issues with cv2
        rotation_matrix = cv2.getRotationMatrix2D((int(x_center), int(y_center)), float(-angle), 1.0)
        rotated_img = cv2.warpAffine(adjusted_image, rotation_matrix, (img.shape[1], img.shape[0]),
                                     flags=cv2.INTER_NEAREST)

        # Re-apply circular mask to the rotated image
        mask_rotated = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.circle(mask_rotated, (int(x_center), int(y_center)), int(new_radius), 255, -1)
        output_img = cv2.bitwise_and(rotated_img, rotated_img, mask=mask_rotated)
    else:
        # no circle found -> return empty image of same shape
        output_img = np.zeros_like(img)

    return output_img
 

def process_simulation_image(img, angle):
    """Optimized rotation using OpenCV warpAffine - much faster than pixel-by-pixel method.
    Uses NEGATED angle to match backward mapping direction and INTER_NEAREST interpolation."""
    
    # Find the center of the image (which is also the center of the inscribed circle)
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    # Radius of the inscribed circle is half the length of the shortest side of the rectangle
    radius = min(center_x, center_y)
    
    # NEGATE the angle because:
    # - Original method uses backward mapping (inverse rotation)
    # - OpenCV uses forward rotation
    # So we need to rotate in the OPPOSITE direction to match!
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    
    # Use INTER_NEAREST to match the original method's int(y), int(x) sampling
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), 
                                  flags=cv2.INTER_NEAREST)
    
    # Create circular mask to keep only pixels within inscribed circle
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Apply mask
    output_img = cv2.bitwise_and(rotated_img, rotated_img, mask=mask)
    
    return output_img

# Define the folders
experimental_folder = EXPERIMENTAL_FOLDER
simulation_folder = SIMULATED_FOLDER

# save by adding _AUG_DUP to the EXPERIMENTAL AND SIMULATED folder names
# extract the base paths and append new folder names
new_experimental_folder = EXPERIMENTAL_FOLDER + '_AUG100_DUP2'
new_simulation_folder = SIMULATED_FOLDER + '_AUG100_DUP2'


# Ensure the new folders exist
os.makedirs(new_experimental_folder, exist_ok=True)
os.makedirs(new_simulation_folder, exist_ok=True)


# Get the file lists
experimental_files = sorted(os.listdir(experimental_folder))
simulation_files = sorted(os.listdir(simulation_folder))

# Verify that each filename matches in both folders
matched_files = set(experimental_files).intersection(simulation_files)

angle_all=np.linspace(0,360,100,endpoint=False)  # rotate by 100 different angles

# Process each pair of images
for filename in sorted(matched_files):
    
    print(filename)
    
    # Read the images ONCE per file (moved outside the angle loop to avoid redundant loading)
    exp_img = cv2.imread(os.path.join(experimental_folder, filename))
    sim_img = cv2.imread(os.path.join(simulation_folder, filename))
     
    for angle in angle_all:
        # Crop and rotate the experimental image
        rotated_exp_img = crop_and_rotate_experimental(exp_img, angle)

        # Process and rotate the simulation image
        rotated_sim_img = process_simulation_image(sim_img, angle)

        # Save the new images in the new folders with the angle in the filename
        new_exp_filename = f"{os.path.splitext(filename)[0]}_rot{angle}{os.path.splitext(filename)[1]}"
        new_sim_filename = f"{os.path.splitext(filename)[0]}_rot{angle}{os.path.splitext(filename)[1]}"

        cv2.imwrite(os.path.join(new_experimental_folder, new_exp_filename), rotated_exp_img)
        cv2.imwrite(os.path.join(new_simulation_folder, new_sim_filename), rotated_sim_img)
        

    # print("Processed and wrote first file")
    # break

print("Processing completed.")
