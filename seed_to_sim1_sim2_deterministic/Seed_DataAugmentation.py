
import os
import numpy as np
import cv2


# Now augment for the seed folder

# augment each image in the seed folder with 3.6 degrees rotations like simulation images
# Also for each seed image, resize to 256x256 pixels so that rotation logic works. 




def process_simulation_image(img, angle):
    # Find the center of the image (which is also the center of the inscribed circle)

    # img=cv2.imread(img)
   

    theta = angle # Replace with your desired rotation angle # taskID varies bw 1-100
    theta_rad = np.deg2rad(theta)

    # resize the image to 256x256 pixels
    img= cv2.resize(img, (256, 256))

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


augmentation_perimage= 100


seed_folder = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input/'
experiment_folder_justchecking= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL'
new_seed_folder= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG'


# Ensure the new folder exists
os.makedirs(new_seed_folder, exist_ok=True)


# select only x number of images from the simulation folder

matched_files_total = set(sorted(os.listdir(seed_folder))).intersection(sorted(os.listdir(experiment_folder_justchecking)))

 # get only the required number of unique samples
print(f"Number of files to process: {len(matched_files_total)}")
angle_all= np.linspace (0,360-3.6, num=augmentation_perimage, endpoint=True)

# Process each pair of images
for filename in matched_files_total:
    
    print(filename)

    # Select a random angle between 0 and 360 degrees 
    # angle = random.randint(0, 359)
    for angle in angle_all:

        # Read the simulation image
        seed_img = cv2.imread(os.path.join(seed_folder, filename))

        # Process and rotate the simulation image
        rotated_seed_img = process_simulation_image(seed_img, angle)
       

        # Save the new images in the new folders with the angle in the filename
        new_seed_filename = f"{os.path.splitext(filename)[0]}_rot{angle}{os.path.splitext(filename)[1]}"  # saving name and extension

       
        cv2.imwrite(os.path.join(new_seed_folder, new_seed_filename), rotated_seed_img)
        
print("Processing completed.")




