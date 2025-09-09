# libraries
import os
import numpy as np
import cv2

# Make folders for diferent data sizes with augmentation, total to 30k samples each
# unique number of samples before augmentation: 100,200,400, 800,1600

# task ID is the number of parallel 

taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])  # Get the task ID from the environment variable, default to 1 if not set


def process_simulation_image(img, angle):
    # Find the center of the image (which is also the center of the inscribed circle)

    # img=cv2.imread(img)
   

    theta = angle 
    theta_rad = np.deg2rad(theta)

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


unique_samples= [10,20,40,60,80,100,200,400,800,1600]
data_sizes= 40000
augmentation_perimage= [int(data_sizes/x) for x in unique_samples]


simulation_folder_1 = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/intermediate/Tp3'
simulation_folder_2= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/complex/Tp3'
new_simulation_folder_1_base= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Data_augmentation_datademand_20250819_intermediate_Tp3'
new_simulation_folder_2_base= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Data_augmentation_datademand_20250819_complex_Tp3'

# Ensure the new folder exists
os.makedirs(new_simulation_folder_1_base, exist_ok=True)
os.makedirs(new_simulation_folder_2_base, exist_ok=True)
new_simulation_folder_1= os.path.join(new_simulation_folder_1_base, f'Augmented_{data_sizes}_uniquesamples_{unique_samples[taskID-1]}')
new_simulation_folder_2= os.path.join(new_simulation_folder_2_base, f'Augmented_{data_sizes}_uniquesamples_{unique_samples[taskID-1]}')
os.makedirs(new_simulation_folder_1, exist_ok=True)
os.makedirs(new_simulation_folder_2, exist_ok=True)
# select only x number of images from the simulation folder

matched_files_total = set(sorted(os.listdir(simulation_folder_1))).intersection(sorted(os.listdir(simulation_folder_2)))

matched_files_subset = sorted(matched_files_total)[:unique_samples[taskID-1]]  # get only the required number of unique samples
print(f"Number of simulation files to process: {len(matched_files_subset)}")
angle_all= np.linspace (0,360, num=augmentation_perimage[taskID-1], endpoint=False)

# Process each pair of images
for filename in matched_files_subset:
    
    print(filename)

    # Select a random angle between 0 and 360 degrees 
    # angle = random.randint(0, 359)
    for angle in angle_all:

        # Read the simulation image
        sim_img_1 = cv2.imread(os.path.join(simulation_folder_1, filename))
        sim_img_2 = cv2.imread(os.path.join(simulation_folder_2, filename))
        # Process and rotate the simulation image
        rotated_sim_img_1 = process_simulation_image(sim_img_1, angle)
        rotated_sim_img_2 = process_simulation_image(sim_img_2, angle)

        # Save the new images in the new folders with the angle in the filename
        new_sim_filename = f"{os.path.splitext(filename)[0]}_rot{angle}{os.path.splitext(filename)[1]}"  # saving name and extension

       
        cv2.imwrite(os.path.join(new_simulation_folder_1, new_sim_filename), rotated_sim_img_1)
        cv2.imwrite(os.path.join(new_simulation_folder_2, new_sim_filename), rotated_sim_img_2)

print("Processing completed.")


