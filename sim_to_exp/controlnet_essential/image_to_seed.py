import os
import cv2
import numpy as np

def image_to_seed(input_image, num_dots=50, min_area=40):
    """
    Convert an input grayscale image into a seed image by extracting 
    equispaced dots along detected contours.
    
    Parameters:
        input_image (np.ndarray): Grayscale input image.
        num_dots (int): Total number of dots to distribute along all contours.
        min_area (int): Minimum area to consider for contours (not used in this version).
        
    Returns:
        np.ndarray: 3-channel (BGR) seed image.
    """
    # # Resize image to 256x256
    img = cv2.resize(input_image, (256, 256))
    
    # Convert to grayscale if it's not already.
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    img=img_gray

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_new = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Edge detection using Canny.
    edges = cv2.Canny(img_new, threshold1=100, threshold2=200)
    
    # Find contours and close them by connecting the last point to the first.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    closed_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            first_point = contour[0]
            first_point_reshaped = first_point.reshape((1, first_point.shape[0], first_point.shape[1]))
            closed_contour = np.vstack((contour, first_point_reshaped))
            closed_contours.append(closed_contour)
    contours = closed_contours

    # Calculate total arc length of all contours.
    total_length = sum(cv2.arcLength(contour, True) for contour in contours) if contours else 0
    if total_length == 0:
        # If no contours are detected, return a blank 3-channel image.
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    dot_positions = []
    # Distribute dots along each contour proportionally to its arc length.
    for contour in contours:
        length = cv2.arcLength(contour, True)
        contour_dots = max(1, int((length / total_length) * num_dots))
        num_points = contour_dots
        arc_length = cv2.arcLength(contour, True)
        interval = arc_length / num_points if num_points > 0 else arc_length
        points = []
        accumulated_distance = 0
        for i in range(len(contour) - 1):
            p1 = contour[i][0]
            p2 = contour[i+1][0]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            while distance > 0 and accumulated_distance + distance >= interval:
                ratio = (interval - accumulated_distance) / distance
                x = int((1 - ratio) * p1[0] + ratio * p2[0])
                y = int((1 - ratio) * p1[1] + ratio * p2[1])
                points.append((x, y))
                accumulated_distance -= interval
            accumulated_distance += distance
        contour_points = points[:num_points]
        dot_positions.extend(contour_points)
    
    # Create a blank image and plot the dots.
    dot_image = np.zeros_like(img)
    for x, y in dot_positions:
        dot_image[y, x] = 255  # Draw dot (white pixel)

    # Convert to a 3-channel image to match Gradio requirements.
    dot_image_color = cv2.cvtColor(dot_image, cv2.COLOR_GRAY2BGR)
    return dot_image_color