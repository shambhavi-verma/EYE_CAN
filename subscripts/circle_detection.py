import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_exposure(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_shadows(image, factor=1.0):
    # Adjust shadows by applying gamma correction on the lower intensities
    shadows = np.clip(255.0 * (image / 255.0) ** factor, 0, 255).astype(np.uint8)
    return shadows

def adjust_highlights(image, factor=1.0):
    # Adjust highlights by appl ying gamma correction on the inverted image
    highlights = np.clip(255.0 * ((255 - image) / 255.0) ** factor, 0, 255).astype(np.uint8)
    # Invert back to normal highlights
    highlights = 255 - highlights
    return highlights

def adjust_saturation(image, factor):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Convert the saturation channel to float32 to prevent overflow during multiplication
    hsv_image = hsv_image.astype(np.float32)
    
    # Scale the saturation channel by the given factor
    hsv_image[:, :, 1] *= factor
    
    # Clip the saturation values to be in valid range [0, 255]
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    
    # Convert back to uint8
    hsv_image = hsv_image.astype(np.uint8)
    
    # Convert the image back to BGR color space
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

def adjust_contrast_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE to the L-channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    
    # Merge the channels back together
    enhanced_lab_image = cv2.merge((l_channel_clahe, a_channel, b_channel))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def increase_whites(image, threshold=200, increment=30):
    # Ensure the increment is within a valid range
    increment = np.clip(increment, 0, 255)

    # Convert the image to the LAB color space to separate the lightness
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Increase the intensity of light regions (whites) based on the threshold
    l_channel = np.where(l_channel > threshold, np.clip(l_channel + increment, 0, 255), l_channel)

    # Merge the channels back
    enhanced_lab_image = cv2.merge((l_channel, a_channel, b_channel))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

# Load the image
image_path = 'pupil.png'
img = cv2.imread(image_path)

# Apply a Gaussian blur to reduce noise and improve circle detection
whitess = increase_whites(img)
blurred_img = adjust_contrast_clahe(whitess, clip_limit=4.0, tile_grid_size=(10, 10))

blurred_img = cv2.GaussianBlur(blurred_img, (9, 9), 2)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Use Hough Circle Transform to detect circles
circles = cv2.HoughCircles(gray_img, 
                           cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=70, param2=16, minRadius=50, maxRadius=100)
# Check if any circles were detected
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

# Show the result
cv2.imshow("blurr", gray_img)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()