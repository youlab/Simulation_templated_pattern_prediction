import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import tempfile
import os
    


""" 
This function processes raw experimental patterns and processes them in the desired color format

"""

def preprocess_experimental_initialstage_mod(img_path, img_length=256, img_width=256):

    """
    Preprocess single image in color from raw experimental data.

    Args:
        img_path: Input image path 
        img_length, img_width: Output dimensions
        use_gentle: Use gentler segmentation for noisy images

    Returns: PIL.Image: Color and Cropped image of Experimental data. Cropped exterior is a white masked circle. 
    """

    img = cv2.imread(img_path)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in the image using Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles method
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=20, maxRadius=600)

    # Assuming the first detected circle is the plate (adjust accordingly)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_center, y_center, radius) in circles:
            # Define a new radius for the mask that is 20 pixels smaller than the detected radius
            new_radius = radius - 82

            # Create a mask where all values are set to zeros (black)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)   

            # Draw a filled white circle on the mask where the new, smaller circle is
            cv2.circle(mask, (x_center, y_center), new_radius, (255, 255, 255), -1)

            # Apply the mask to the original image (set pixels outside the new circle to white)
            img_masked = cv2.bitwise_and(img, img, mask=mask)

            # Adjust contrast and brightness
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 50    # Brightness control (0-100)
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)


            # Convert BGR to RGB for plotting
            img_masked_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

            # resize both the adjusted image and the mask
            img_resized = cv2.resize(img_masked_rgb, (img_length, img_width))
            mask_resized = cv2.resize(mask,(img_length, img_width),interpolation=cv2.INTER_NEAREST)

            # re-mask the resized image to force a black border outside the circle
            img_masked_resized = cv2.bitwise_and(img_resized,img_resized,mask=mask_resized)

            # Convert black border to white border (quick fix without changing bitwise operations)
            # Create inverse mask for border areas
            border_mask = mask_resized == 0
            img_masked_resized[border_mask] = [255, 255, 255]  # Set border to white

            return img_masked_resized

    else:
        print("No circles were found")
        return None



"""Scale latents for visualization"""

def scale_latents(latent):
    # Normalize latent for visualization
    latent = latent - latent.min()
    latent = latent / latent.max()
    return latent


def grayfordisplay(img_path, img_length=256, img_width=256, img_type='sim'):
    """
    Preprocess single image Do 
    1. Invert colors, white to black and black to white
    2. Convert black to gray for better visualization
    3. Resize to desired dimensions

    Args:
        img_path: Input image path 
        img_length, img_width: Output dimensions
        use_gentle: Use gentler segmentation for noisy images
    Returns: cv2 image: Processed image of Experimental data.

    """
    
    img=cv2.imread(img_path, cv2.IMREAD_COLOR)
    # convert to RGB
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_inverse = cv2.bitwise_not(np.array(img))
    

    # change artificial color to grey for visualization purposes
    # RGB color code : (70,70,70)
    img_inverse[np.all(img_inverse == [0,0,0], axis=-1)] = [152,152,152]

    if img_type=='seed':
        return img_inverse # for seed images, just invert and change black to gray resize causing weird artifacts

    # remove the small border artifacts by adding a white border
    # Set all channels including alpha to 255 for opaque white borders
    img_inverse[0:2, :] = [255, 255, 255]
    img_inverse[-2:, :] = [255, 255, 255]
    img_inverse[:, 0:2] = [255, 255, 255]
    img_inverse[:, -2:] = [255, 255, 255]

    img_inverse = cv2.resize(img_inverse, (img_length, img_width))  # resize after doing operations

    return img_inverse    

def preprocess_simulation_output_data(img_path, start_index, end_index, img_filenames=None, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,  img_length=256, img_width=256):
    """
    Preprocess and crops simulation output images in grayscale from raw simulation data.

    Args:
        img_path: Input image path 
        top_crop, bottom_crop, left_crop, right_crop: Number of pixels to crop from each side
        start_index, end_index: Indices to crop the image array
        path_img: Path to the folder containing images
        img_length, img_width: Output dimensions   
    Returns: PIL.Image: Color and Cropped image of Simulation data. 
    """
    count = 0
    output_data = []

    # for simcorrtoexp, we have a list of sorted filenames to avoid data leakage
    
    if img_filenames is not None:
        img_filenames_o = img_filenames
    else: 
        img_filenames_o = sorted(os.listdir(img_path))
    
    

    # Ensure end_index does not exceed the number of available images
    if end_index == -1:
        end_index = len(img_filenames_o)
    else:
        end_index = min(end_index, len(img_filenames_o))

    for img in img_filenames_o[start_index:end_index]:
        img_array_o = cv2.imread(os.path.join(img_path, img), cv2.IMREAD_GRAYSCALE)

        new_height = img_array_o.shape[0] - (top_crop + bottom_crop)
        new_width = img_array_o.shape[1] - (left_crop + right_crop)

        new_array_o = img_array_o[top_crop:top_crop + new_height, left_crop:left_crop + new_width]

        new_array_o = cv2.resize(new_array_o, (img_length, img_width))

        # Convert images to gray on light background for better visualization
        img_inverse = cv2.bitwise_not(np.array(new_array_o))
        
        #  Change artificial color to gray for visualization
        #  Extra thresholding step for predicted images to remove noise
        img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
        #  Change black to gray (for grayscale images)
        img_inverse[img_inverse == 0] = 152

        output_data.append([img_inverse])
        count = count + 1
        if count >= (end_index - start_index):
            break
    return output_data


def preprocess_simulation_input_data(img_path, start_index, end_index,img_filenames=None, top_crop=0, bottom_crop=0, left_crop=3, right_crop=2):
    
    """
    Preprocess and crops simulation input images in grayscale from raw simulation data.

    Args:
        img_path: Input image path 
        top_crop, bottom_crop, left_crop, right_crop: Number of pixels to crop from each side
        start_index, end_index: Indices to crop the image array
        path_img: Path to the folder containing images
        img_length, img_width: Output dimensions   
    Returns: PIL.Image: Color and Cropped image of Simulation data. Cropped exterior is a white masked circle. 
    """

    # note: no resizing for input seed images, images are already img_length=32, img_width=32
    count=0
    input_data=[]

    if img_filenames is not None:
        img_filenames_i = img_filenames
    else: 
        img_filenames_i = sorted(os.listdir(img_path))

    # Ensure end_index does not exceed the number of available images
    end_index = min(end_index, len(img_filenames_i))
    
    for img in img_filenames_i[start_index:end_index]:
        img_array_i=cv2.imread(os.path.join(img_path,img),cv2.IMREAD_GRAYSCALE)

        new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
        new_width = img_array_i.shape[1] - (left_crop + right_crop)
        new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

        # Skip OTSU thresholding for seed images - they're already binary
        (T, new_array_i) = cv2.threshold(new_array_i, 0, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

        # Convert images to gray on light background for better visualization
        img_inverse = cv2.bitwise_not(np.array(new_array_i))
        
        #  Change artificial color to gray for visualization
        #  Extra thresholding step for predicted images to remove noise
        img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
        #  Change black to gray (for grayscale images)
        img_inverse[img_inverse == 0] = 152

        input_data.append([img_inverse])
        count=count+1
        if count >= (end_index - start_index):
            break
    return input_data

def process_tensor_batch_with_grayfordisplay(tensor_batch, img_length=256, img_width=256, img_type='sim'):
    """
    Process a batch of torch tensors using grayfordisplay logic directly
    
    Args:
        tensor_batch: torch.Tensor of shape (B, C, H, W) where C can be 1 or 3
        img_length, img_width: Output dimensions
        img_type: 'sim' or 'seed'
    
    Returns:
        torch.Tensor: Processed images in same batch format as input (B, C, H, W)
    """
    batch_size, channels = tensor_batch.shape[0], tensor_batch.shape[1]
    processed_images = []
    
    for i in range(batch_size):
        # Extract single image from batch: (C, H, W)
        single_tensor = tensor_batch[i].cpu().numpy()
        
        # Handle both 1-channel and 3-channel inputs
        if channels == 1:
            # Convert grayscale to RGB: (1, H, W) -> (H, W, 3)
            img = single_tensor[0]  # Remove channel dimension
            img = np.stack([img, img, img], axis=-1)  # Convert to 3-channel
        else:
            # Convert RGB: (3, H, W) -> (H, W, 3)
            img = single_tensor.transpose(1, 2, 0)
        
        # Ensure proper data type and range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Apply grayfordisplay logic 
     
        # 1. Invert colors
        img_inverse = cv2.bitwise_not(np.array(img))
        
        # 2. Change artificial color to gray for visualization
        # 2.1 Extra thresholding step for predicted images to remove noise
        img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
        # 2.2 Change black to gray
        img_inverse[np.all(img_inverse == [0,0,0], axis=-1)] = [152,152,152]
        
        if img_type != 'seed':
            # 3. Remove border artifacts by adding white border
            img_inverse[0:2, :] = [255, 255, 255]
            img_inverse[-2:, :] = [255, 255, 255]
            img_inverse[:, 0:2] = [255, 255, 255]
            img_inverse[:, -2:] = [255, 255, 255]
            
            # 4. Resize after operations
            img_inverse = cv2.resize(img_inverse, (img_length, img_width))
        
        # Convert back to tensor and normalize to [0, 1]
        processed_tensor = torch.from_numpy(img_inverse.astype(np.float32) / 255.0)
        
        # Convert back to original format
        if channels == 1:
            # Convert back to grayscale: (H, W, 3) -> (1, H, W)
            processed_tensor = processed_tensor.mean(dim=-1, keepdim=True).permute(2, 0, 1)
        else:
            # Convert back to RGB: (H, W, 3) -> (3, H, W)
            processed_tensor = processed_tensor.permute(2, 0, 1)
        
        processed_images.append(processed_tensor)
    
    # Stack back into batch tensor
    processed_batch = torch.stack(processed_images, dim=0)
    
    # Ensure same device as input
    return processed_batch.to(tensor_batch.device)



def load_sorted_filenames(file_path):
    with open(file_path, 'r') as file:
        sorted_filenames = [line.strip() for line in file.readlines()]
    return sorted_filenames


# reads sorted filnames from a text file, to avoid alphanumeric sorting issues and avoid rotation data leakage
def preprocess_experimental_output_data(path_i,start_index,end_index,img_filenames=None,top_crop=25,bottom_crop=25,left_crop=25,right_crop=25,img_length=256,img_width=256):
    """
    Preprocess experimental output images by cropping, resizing, and inverting colors.

    Parameters:
        path_i (str): Directory path containing the images (not raw files, but processed images with the plate being masked)
        start_index (int): Starting index of images to process.
        end_index (int): Ending index of images to process.
        img_filenames (list, optional): List of image filenames to process. Defaults to None.
        top_crop (int, optional): Pixels to crop from the top. Defaults to 25.
        bottom_crop (int, optional): Pixels to crop from the bottom. Defaults to 25.
        left_crop (int, optional): Pixels to crop from the left. Defaults to 25.
        right_crop (int, optional): Pixels to crop from the right. Defaults to 25.
        img_length (int, optional): Desired image height after processing. Defaults to 256.
        img_width (int, optional): Desired image width after processing. Defaults to 256.

    Returns:
        list: List of processed image arrays (Black and White, then inverted with gray background)
    """

    count = 0
    output_data = []

    # for simcorrtoexp, we have a list of sorted filenames to avoid data leakage

    if img_filenames is not None:
        img_filenames_o = img_filenames
    else: 
        img_filenames_o = sorted(os.listdir(path_i))

    # Ensure end_index does not exceed the number of available images
    if end_index == -1:
        end_index = len(img_filenames_o)
    else:
        end_index = min(end_index, len(img_filenames_o))

    for img in img_filenames_o[start_index:end_index]:
 
        img_array_i=cv2.imread(os.path.join(path_i,img),cv2.IMREAD_GRAYSCALE)  # ,cv2.IMREAD_GRAYSCALE removed on 050324

        ########################
        ## added on 030524
        blur = cv2.GaussianBlur(img_array_i,(5,5),0)   
        ret3,th3 = cv2.threshold(blur,110,255,cv2.THRESH_BINARY)
        #########################
        img_array_i=th3

        # resize the array
        img_array_i = cv2.resize(img_array_i, (img_width, img_length))

        new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
        new_width = img_array_i.shape[1] - (left_crop + right_crop)
    
        new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]


        # blurred = cv2.GaussianBlur(img_array_i, (7, 7), 0)
        # (T, img_array_i) = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)   # removing _INV
        new_array_i=cv2.resize(new_array_i,(img_length,img_width))

        img_inverse = cv2.bitwise_not(np.array(new_array_i))
        
        # 2. Change artificial color to gray for visualization
        # 2.1 Extra thresholding step for predicted images to remove noise
        img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
        # 2.2 Change black to gray (for grayscale images)
        img_inverse[img_inverse == 0] = 152

        output_data.append([img_inverse])
        count = count + 1
        if count >= (end_index - start_index):
            break
    return output_data
        
def preprocess_experimental_output_data_rawfiles(path_i,start_index,end_index,img_filenames=None,top_crop=25,bottom_crop=25,left_crop=25,right_crop=25,img_length=256,img_width=256):
    """
    Preprocess experimental output images by applying circle masking (from rawfiles), cropping, resizing, and inverting colors.

    Parameters:
        path_i (str): Directory path containing the images (raw files)
        start_index (int): Starting index of images to process.
        end_index (int): Ending index of images to process.
        img_filenames (list, optional): List of image filenames to process. Defaults to None.
        top_crop (int, optional): Pixels to crop from the top. Defaults to 25.
        bottom_crop (int, optional): Pixels to crop from the bottom. Defaults to 25.
        left_crop (int, optional): Pixels to crop from the left. Defaults to 25.
        right_crop (int, optional): Pixels to crop from the right. Defaults to 25.
        img_length (int, optional): Desired image height after processing. Defaults to 256.
        img_width (int, optional): Desired image width after processing. Defaults to 256.

    Returns:
        list: List of processed image arrays (Black and White, then inverted with gray background)
    """

    count = 0
    output_data = []

    # for simcorrtoexp, we have a list of sorted filenames to avoid data leakage

    if img_filenames is not None:
        img_filenames_o = img_filenames
    else: 
        img_filenames_o = sorted(os.listdir(path_i))

    # Ensure end_index does not exceed the number of available images
    if end_index == -1:
        end_index = len(img_filenames_o)
    else:
        end_index = min(end_index, len(img_filenames_o))

    for img in img_filenames_o[start_index:end_index]:
        # Read image in color first for circle detection
        img_array = cv2.imread(os.path.join(path_i, img), cv2.IMREAD_COLOR)
        
        # Convert the image to grayscale for circle detection
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Find edges in the image using Canny edge detection
        edges = cv2.Canny(img_gray, 100, 200)

        # Find circles in the image using HoughCircles method
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=20, maxRadius=600)

        # Process the image based on circle detection
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Take the first circle found
            x_center, y_center, radius = circles[0]
            
            # Define a new radius for the mask that is smaller than the detected radius
            new_radius = radius - 82

            # Create a mask where all values are set to zeros (black)
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)   

            # Draw a filled white circle on the mask where the new, smaller circle is
            cv2.circle(mask, (x_center, y_center), new_radius, (255, 255, 255), -1)

            # Apply the mask to the original image
            img_masked = cv2.bitwise_and(img_array, img_array, mask=mask)

            # Adjust contrast and brightness
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 50    # Brightness control (0-100)
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

            # Convert to grayscale after masking and adjustment
            img_array_i = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

            # resize both the adjusted image and the mask
            img_resized = cv2.resize(img_array_i, (img_length, img_width))
            mask_resized = cv2.resize(mask, (img_length, img_width), interpolation=cv2.INTER_NEAREST)

            # re-mask the resized image
            img_array_i = cv2.bitwise_and(img_resized, img_resized, mask=mask_resized)
            
        else:
            print(f"No circles were found in {img}, processing without masking")
            # Process without masking if no circles found
            img_array_i = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array_i = cv2.resize(img_array_i, (img_width, img_length))

        ########################
        ## Apply blur and threshold (same as original)
        blur = cv2.GaussianBlur(img_array_i,(5,5),0)   
        ret3,th3 = cv2.threshold(blur,110,255,cv2.THRESH_BINARY)
        #########################
        img_array_i=th3

        # Crop the image
        new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
        new_width = img_array_i.shape[1] - (left_crop + right_crop)
    
        new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

        # Final resize after cropping
        new_array_i=cv2.resize(new_array_i,(img_length,img_width))

        # Invert colors (same as original output format)
        img_inverse = cv2.bitwise_not(np.array(new_array_i))
        
        # Change artificial color to gray for visualization (same as original)
        # Extra thresholding step for predicted images to remove noise
        img_inverse=cv2.threshold(img_inverse, 10, 255, cv2.THRESH_TOZERO)[1]
        # Change black to gray (for grayscale images)
        img_inverse[img_inverse == 0] = 152

        output_data.append([img_inverse])
        count = count + 1
        if count >= (end_index - start_index):
            break
    return output_data
           
   


def preprocess_experimental_backgroundwhite(exp_path,start_index, end_index, img_filenames=None, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
  
    """
    Preprocess a list of images from the given path and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file, assumes experimental image is already processed to exclude petri dish area and that area is black
        top_crop (int): Pixels to crop from the top.---------- 
        bottom_crop (int): Pixels to crop from the bottom.----All are kept 25 for experimental images 
        left_crop (int): Pixels to crop from the left.--------
        right_crop (int): Pixels to crop from the right.------
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized background white image as a numpy array converted to RGB.
    """

    count=0
    output_data=[]

    if img_filenames is not None:
        img_filenames_o = img_filenames
    else: 
        img_filenames_o = sorted(os.listdir(exp_path))

    # Ensure end_index does not exceed the number of available images
    end_index = min(end_index, len(img_filenames_o))

    for img in img_filenames_o[start_index:end_index]:
        # Read image using cv2
        img_array = cv2.imread(os.path.join(exp_path, img), cv2.IMREAD_COLOR)  # Read in color to preserve colony colors

    # convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # resize the array
        img_array = cv2.resize(img_array, (img_width, img_length))

        if img_array is None:
            print(f"Failed to load image at {exp_path}")
            return None

        ########################
        # do some cropping , note do appropriate cropping for simulation images too
        ########################

        new_height = img_array.shape[0] - (top_crop + bottom_crop)
        new_width = img_array.shape[1] - (left_crop + right_crop)

        new_array_i = img_array[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

        # now change black to white for the masked area
        new_array_i[np.all(new_array_i == [0,0,0], axis=-1)] = [255,255,255]  # change black to white

        # resize after cropping
        new_array_i = cv2.resize(new_array_i, (img_width, img_length))

        output_data.append(new_array_i)
        count=count+1
        if count >= (end_index - start_index):
            break

    return output_data


def preprocess_experimental_backgroundwhite_rawfiles(exp_path,start_index, end_index, img_filenames=None, top_crop=25, bottom_crop=25, left_crop=25, right_crop=25,
                                     img_length=256, img_width=256):
  
    """
    Preprocess a single image from the given path and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file, assumes experimental image is already processed to exclude petri dish area and that area is black
        top_crop (int): Pixels to crop from the top.---------- 
        bottom_crop (int): Pixels to crop from the bottom.----All are kept 25 for experimental images 
        left_crop (int): Pixels to crop from the left.--------
        right_crop (int): Pixels to crop from the right.------
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized background white image as a numpy array converted to RGB.
    """

    count=0
    output_data=[]



    if img_filenames is not None:
        img_filenames_o = img_filenames
    else: 
        img_filenames_o = sorted(os.listdir(exp_path))



    # Ensure end_index does not exceed the number of available images
    end_index = min(end_index, len(img_filenames_o))

    for img in img_filenames_o[start_index:end_index]:
        # Read image using cv2
        img_array = cv2.imread(os.path.join(exp_path, img), cv2.IMREAD_COLOR)  # Read in color to preserve colony colors

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Find edges in the image using Canny edge detection
        edges = cv2.Canny(img_gray, 100, 200)

        # Find circles in the image using HoughCircles method
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=20, maxRadius=600)

        # Process the image based on circle detection
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Take the first circle found
            x_center, y_center, radius = circles[0]
            
            # Define a new radius for the mask that is smaller than the detected radius
            new_radius = radius - 82

            # Create a mask where all values are set to zeros (black)
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)   

            # Draw a filled white circle on the mask where the new, smaller circle is
            cv2.circle(mask, (x_center, y_center), new_radius, (255, 255, 255), -1)

            # Apply the mask to the original image
            img_masked = cv2.bitwise_and(img_array, img_array, mask=mask)

            # Adjust contrast and brightness
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 50    # Brightness control (0-100)
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

            # Convert BGR to RGB for plotting
            img_masked_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

            # resize both the adjusted image and the mask
            img_resized = cv2.resize(img_masked_rgb, (img_length, img_width))
            mask_resized = cv2.resize(mask, (img_length, img_width), interpolation=cv2.INTER_NEAREST)

            # re-mask the resized image to force a black border outside the circle
            img_masked_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask_resized)

            # do some cropping
            new_height = img_masked_resized.shape[0] - (top_crop + bottom_crop)
            new_width = img_masked_resized.shape[1] - (left_crop + right_crop)

            img_masked_resized = img_masked_resized[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

            # now change black to white for the masked area
            img_masked_resized[np.all(img_masked_resized == [0,0,0], axis=-1)] = [255,255,255]  # change black to white

            # resize after cropping
            img_masked_resized = cv2.resize(img_masked_resized, (img_width, img_length))

            output_data.append(img_masked_resized)
            
        else:
            print(f"No circles were found in {img}, skipping this image")
            # Create a simple fallback processed image
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (img_length, img_width))
            
            # do basic cropping
            new_height = img_resized.shape[0] - (top_crop + bottom_crop)
            new_width = img_resized.shape[1] - (left_crop + right_crop)
            img_cropped = img_resized[top_crop:top_crop+new_height, left_crop:left_crop+new_width]
            img_final = cv2.resize(img_cropped, (img_width, img_length))
            
            output_data.append(img_final)
        
        count = count + 1
        if count >= (end_index - start_index):
            break

    return output_data

