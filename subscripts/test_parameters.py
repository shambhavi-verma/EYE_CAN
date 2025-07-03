import cv2
import numpy as np

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

def adjust(img_path):

    img = cv2.imread(img_path)

    if img is None:
        print(f"Unable to load image from {img_path}")
        return

    img = cv2.resize(img, (1080,720))

    adjusted = adjust_saturation(img, 1.4)
    adjusted = adjust_brightness(adjusted, 20)
    adjusted = adjust_exposure(adjusted, 1.3)
    adjusted = adjust_shadows(adjusted, 0.7)
    adjusted_img = adjust_highlights(adjusted, 1.4)    

    return adjusted_img

def calculate_mean(list_of_lists):
    array = np.array(list_of_lists)
    mean_values = np.mean(array, axis=0)
    return mean_values.tolist()

# Load image
image = cv2.imread("eye.jpg")
adjusted = image

##Adjust brightness
adjusted = adjust_saturation(image, 1.8)

adjusted = adjust_brightness(adjusted, 6)

adjusted = adjust_exposure(adjusted, 0.7)

adjusted = adjust_shadows(adjusted, 0.7)

adjusted = adjust_highlights(adjusted, 1.4)

adjusted = cv2.GaussianBlur(adjusted, (5,5), 6)

canny = cv2.Canny(adjusted, 125, 175)
        
cv2.imshow("new_img", canny)
cv2.imshow("img", adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()