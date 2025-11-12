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

    # similar code as the some of the preprocessing functions

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in the image using Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles method
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=20, maxRadius=600)
    
    # print(circles)

    # Assuming the first detected circle is the plate (adjust accordingly)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Take the first circle found
        # Assuming first circle detected by Hough transform is the plate
        x_center, y_center, radius = circles[0]
        
        # Define a new radius for the mask that is 20 pixels smaller than the detected radius
        new_radius = radius -82
        # print(new_radius)

        # Create a mask where all values are set to zero (black)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Draw a filled white circle on the mask where the new, smaller circle is
        cv2.circle(mask, (x_center, y_center), new_radius, (255, 255, 255), -1)

        # Apply the mask to the original image (set pixels outside the new circle to black)
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        
        # Adjust contrast and brightness
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 50    # Brightness control (0-100)
        adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

        
        # Specify the rotation angle theta in degrees
        theta = angle
        theta_rad = np.deg2rad(theta)

        # Create an output image (initially black or any background color you prefer)
        output_img = np.zeros_like(img)

        # Iterate over each pixel in the output image
        for x_prime in range(img.shape[1]):
            for y_prime in range(img.shape[0]):
                # Apply the inverse rotation transformation (backward mapping)
                x = np.cos(theta_rad) * (x_prime - x_center) + np.sin(theta_rad) * (y_prime - y_center) + x_center
                y = -np.sin(theta_rad) * (x_prime - x_center) + np.cos(theta_rad) * (y_prime - y_center) + y_center

                # Check if the source pixel (x, y) is within the new circle's boundaries
                if (x - x_center)**2 + (y - y_center)**2 <= new_radius**2:
                    # Check if it's inside the image
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        # Sample the pixel from the masked image to the output image
                        output_img[y_prime, x_prime] = adjusted_image[int(y), int(x)]

        
    else:
        print("No circles were found")

    return output_img
 

def process_simulation_image(img, angle):
    # Find the center of the image (which is also the center of the inscribed circle)

    # img=cv2.imread(img)

    theta = angle # Replace with your desired rotation angle
    theta_rad = np.deg2rad(theta)

    # List all image files in the folder
    # image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image file
    # for image_file in image_files:
        # Read the image
    # img = cv2.imread(os.path.join(folder_path, filepath))

    # Find the center of the image (which is also the center of the inscribed circle)
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    # Radius of the inscribed circle is half the length of the shortest side of the rectangle
    radius = min(center_x, center_y)

    # Create an output image (initially black or any background color you prefer)
    output_img = np.zeros_like(img)

    # Perform backward mapping rotation for points within the inscribed circle
    for x_prime in range(img.shape[1]):
        for y_prime in range(img.shape[0]):
            # Apply the inverse rotation transformation (backward mapping)
            x = np.cos(theta_rad) * (x_prime - center_x) + np.sin(theta_rad) * (y_prime - center_y) + center_x
            y = -np.sin(theta_rad) * (x_prime - center_x) + np.cos(theta_rad) * (y_prime - center_y) + center_y

            # Check if the source pixel (x, y) is within the inscribed circle's boundaries
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                # Check if it's inside the image
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    # Sample the pixel from the original image to the output image
                    output_img[y_prime, x_prime] = img[int(y), int(x)]
    
    return output_img

# Define the folders
experimental_folder = EXPERIMENTAL_FOLDER
simulation_folder = SIMULATED_FOLDER

# save by adding _AUG_DUP to the EXPERIMENTAL AND SIMULATED folder names
# extract the base paths and append new folder names
new_experimental_folder = EXPERIMENTAL_FOLDER + '_AUG100_DUP'
new_simulation_folder = SIMULATED_FOLDER + '_AUG100_DUP'


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
     
    for angle in angle_all:
        # Read the experimental image
        exp_img = cv2.imread(os.path.join(experimental_folder, filename))
        # Crop and rotate the experimental image
        rotated_exp_img = crop_and_rotate_experimental(exp_img, angle)

        # Read the simulation image
        sim_img = cv2.imread(os.path.join(simulation_folder, filename))
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
