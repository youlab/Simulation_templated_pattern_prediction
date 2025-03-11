from PIL import Image
import numpy as np
import cv2



def preprocess_experimental(exp_path, top_crop=30, bottom_crop=30, left_crop=31, right_crop=30,
                                     img_length=256, img_width=256):
    """
    Preprocess a single image from the given path and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file.
        top_crop (int): Pixels to crop from the top.
        bottom_crop (int): Pixels to crop from the bottom.
        left_crop (int): Pixels to crop from the left.
        right_crop (int): Pixels to crop from the right.
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        Resized grasyscale image
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path,cv2.IMREAD_GRAYSCALE) 
    
    if img_array is None:
        print(f"Failed to load image at {exp_path}")
        return None

    ########################
    ## added on 030524
    blur = cv2.GaussianBlur(img_array,(5,5),0)   
    ret3,th3 = cv2.threshold(blur,110,255,cv2.THRESH_BINARY)
    #########################
    img_array=th3

    new_height = img_array.shape[0] - (top_crop + bottom_crop)
    new_width = img_array.shape[1] - (left_crop + right_crop)

    new_array_i = img_array[top_crop:top_crop+new_height, left_crop:left_crop+new_width]


    # blurred = cv2.GaussianBlur(img_array_i, (7, 7), 0)
    # (T, img_array_i) = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)   # removing _INV
    experimental_image_resized=cv2.resize(new_array_i,(img_length,img_width))
    return experimental_image_resized
    

    


def preprocess_simulation(sim_path, top_crop=30, bottom_crop=30, left_crop=31, right_crop=30,
                                     img_length=256, img_width=256):
    
    # Read image using cv2
    img_array = cv2.imread(sim_path)
    
    if img_array is None:
        print(f"Failed to load image at {sim_path}")
        return None

    #convert to grayscale
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    #note: modification on 02 11 2025 to check how it will look with uncropped images

    # top_crop=0
    # bottom_crop=0
    # left_crop=0
    # right_crop=0

    # un modified back , it works, by default using cropping for now
    # for reference see /hpc/dctrl/ks723/inference/v20250211_2231/

    new_height = img_gray.shape[0] - (top_crop + bottom_crop)
    new_width = img_gray.shape[1] - (left_crop + right_crop)

    new_img = img_gray[top_crop:top_crop+new_height, left_crop:left_crop+new_width]
    simulated_image_resized=cv2.resize(new_img,(img_length,img_width))
    return simulated_image_resized



def preprocess_seed(sim_path, top_crop=0, bottom_crop=0, left_crop=0, right_crop=0,
                                     img_length=256, img_width=256):
    
    # Read image using cv2
    img_array = cv2.imread(sim_path)
    
    if img_array is None:
        print(f"Failed to load image at {sim_path}")
        return None

    #convert to grayscale
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    #note: modification on 02 11 2025 to check how it will look with uncropped images

    # top_crop=0
    # bottom_crop=0
    # left_crop=0
    # right_crop=0

    # un modified back , it works, by default using cropping for now
    # for reference see /hpc/dctrl/ks723/inference/v20250211_2231/

    new_height = img_gray.shape[0] - (top_crop + bottom_crop)
    new_width = img_gray.shape[1] - (left_crop + right_crop)

    new_img = img_gray[top_crop:top_crop+new_height, left_crop:left_crop+new_width]
    seed_image_resized=cv2.resize(new_img,(img_length,img_width))
    return seed_image_resized


def preprocess_experimental_fromscratch(exp_path, top_crop=30, bottom_crop=30, left_crop=31, right_crop=30,
                                     img_length=256, img_width=256):
    """
    Preprocess a single image from the given path and return a PIL-compatible output.
    
    Parameters:
        exp_path (str): Path to the experimental image file.
        top_crop (int): Pixels to crop from the top.
        bottom_crop (int): Pixels to crop from the bottom.
        left_crop (int): Pixels to crop from the left.
        right_crop (int): Pixels to crop from the right.
        img_length (int): Desired output image height.
        img_width (int): Desired output image width.
        
    Returns:
        PIL.Image: Processed image in PIL format.
    """
    # Read image using cv2
    img_array = cv2.imread(exp_path)
    
    if img_array is None:
        print(f"Failed to load image at {exp_path}")
        return None

    # Convert to grayscale for edge and circle detection
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=600)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_center, y_center, radius) in circles:
            new_radius = radius - 82
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x_center, y_center), new_radius, 255, -1)

            # Apply the mask to the original image
            img_masked = cv2.bitwise_and(img_array, img_array, mask=mask)

            # Adjust contrast and brightness
            alpha = 1.5  # Contrast control
            beta = 50    # Brightness control
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

            # Convert to grayscale and apply thresholding
            img_gray_adj = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(img_gray_adj, (5, 5), 0)
            _, img_thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)

            # Crop and resize the image
            new_height = img_thresh.shape[0] - (top_crop + bottom_crop)
            new_width = img_thresh.shape[1] - (left_crop + right_crop)
            img_cropped = img_thresh[top_crop:top_crop + new_height, left_crop:left_crop + new_width]
            img_resized = cv2.resize(img_cropped, (img_length, img_width))

            # # Convert the final output to PIL format
            # final_image_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB))
            return img_resized
    else:
        print("No circles were found")
        return None
    
def preprocess_vanillaresize(sim_path, img_length=256, img_width=256):
    
    # Read image using cv2
    img_array = cv2.imread(sim_path)

    if img_array is None:
        return None
    if len(img_array.shape) == 3 and img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    
    seed_image_resized=cv2.resize(img_array,(img_length,img_width))
    return seed_image_resized


def preprocess_experimental_initialstage(img_path, img_length=256,img_width=256):

    img = cv2.imread(img_path)

    # Convert the image to grayscale
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
        for (x_center, y_center, radius) in circles:
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

            

            # Convert BGR images to RGB for plotting with matplotlib
            img_masked_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
            # output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

            # output_img_grayscale=cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(output_img_grayscale,(5,5),0)   
            # ret3,th3 = cv2.threshold(blur,110,255,cv2.THRESH_BINARY)
            # ########################
            # img_array_i=th3


            img_masked_resized=cv2.resize(img_masked_rgb,(img_length,img_width))
            return img_masked_resized
    else:
        print("No circles were found")
        return None








