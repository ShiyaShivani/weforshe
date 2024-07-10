import cv2
import os
import numpy as np

def add_alpha_channel(image):
    """Convert BGR image to BGRA by adding an alpha channel."""
    b_channel, g_channel, r_channel = cv2.split(image)
    alpha_channel = np.ones(b_channel.shape, dtype_b_channel.dtype) * 255  # Create a dummy alpha channel
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def overlay_clothing(frame, clothing_img, keypoints):
    if clothing_img.shape[2] == 3:  # Check if the image has no alpha channel
        clothing_img = add_alpha_channel(clothing_img)
    
    # Calculate the width of the shoulders
    shoulder_width = int(np.linalg.norm(np.array(keypoints['RIGHT_SHOULDER']) - np.array(keypoints['LEFT_SHOULDER'])))
    
    # Scale the clothing image to match the shoulder width
    scale_factor = 2.0  # Increased scale factor for larger clothing
    clothing_aspect_ratio = clothing_img.shape[0] / clothing_img.shape[1]
    new_width = int(shoulder_width * scale_factor)
    new_height = int(new_width * clothing_aspect_ratio)

    # Ensure new dimensions fit within frame
    max_height = frame.shape[0] - keypoints['LEFT_SHOULDER'][1]
    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height / clothing_aspect_ratio)

    resized_clothing = cv2.resize(clothing_img, (new_width, new_height))
    
    # Position the clothing image to align with the shoulders
    shoulder_midpoint_x = int((keypoints['LEFT_SHOULDER'][0] + keypoints['RIGHT_SHOULDER'][0]) / 2)
    overlay_x = shoulder_midpoint_x - new_width // 2
    overlay_y = keypoints['LEFT_SHOULDER'][1] - new_height // 4  # Adjust this value if needed to better position the clothing
    
    # Ensure the overlay region is within frame bounds
    overlay_x = max(0, overlay_x)
    overlay_y = max(0, overlay_y)

    if overlay_x + resized_clothing.shape[1] > frame.shape[1]:
        resized_clothing = resized_clothing[:, :frame.shape[1] - overlay_x, :]
    if overlay_y + resized_clothing.shape[0] > frame.shape[0]:
        resized_clothing = resized_clothing[:frame.shape[0] - overlay_y, :]
    
    # Overlay the clothing image
    for c in range(0, 3):
        frame[overlay_y:overlay_y+resized_clothing.shape[0], overlay_x:overlay_x+resized_clothing.shape[1], c] = \
            resized_clothing[:, :, c] * (resized_clothing[:, :, 3] / 255.0) + \
            frame[overlay_y:overlay_y+resized_clothing.shape[0], overlay_x:overlay_x+resized_clothing.shape[1], c] * (1.0 - resized_clothing[:, :, 3] / 255.0)

def load_clothing_images(folder_path):
    clothing_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_UNCHANGED)
            if img is not None:
                clothing_images.append(img)
    return clothing_images
