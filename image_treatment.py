from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import sklearn
import sklearn.cluster
import itertools
import statistics
import cv2
import scipy

def edge_detection(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Apply the Sobel filters for horizontal and vertical edge detection
    sobel_x = cv2.Sobel(img[:,:,1], cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edge detection
    sobel_y = cv2.Sobel(img[:,:,1], cv2.CV_64F, 0, 1, ksize=3)  # Vertical edge detection

    # Calculate the magnitude of the gradient
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the result to an 8-bit scale
    edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display the edge magnitude image
    return edge_magnitude

def binary_one(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_red = np.array([10, 150, 50])
    upper_red = np.array([255, 255, 255])

    # Create a binary mask for the red color within the specified range
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Apply the mask to the original image to segment the red regions
    red_segmented = cv2.bitwise_and(img, img, mask=mask)
    _, binary_image = cv2.threshold(red_segmented, red_segmented.mean(), 255, cv2.THRESH_BINARY)
    return binary_image

def binary_two(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define a more specific range for red in HSV color space
    lower_red = np.array([0, 100, 100])   # Lower red hue
    upper_red = np.array([30, 255, 255])  # Upper red hue

    # Create a binary mask for the red color within the specified range
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Apply the mask to the original image to segment the red regions
    red_segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    # Convert the result back to RGB for display
    red_segmented_rgb = cv2.cvtColor(red_segmented, cv2.COLOR_BGR2RGB)

    return red_segmented_rgb

def binary_three(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([100, 100, 100])
    upper_red = np.array([255, 255, 255])

    # Create a binary mask for the red color within the specified range
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Apply the mask to the original image to segment the red regions
    red_segmented = cv2.bitwise_and(img, img, mask=mask)
    _, binary_image = cv2.threshold(red_segmented, red_segmented.mean(), 255, cv2.THRESH_BINARY)
    return binary_image

def equalized(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    equalized_channels = [cv2.equalizeHist(channel) for channel in cv2.split(img)]
    equalized_img = cv2.merge(equalized_channels)

    return equalized_img

def blur(img_bgr, kernel_size=5):
    return cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), 0)

# https://medium.com/featurepreneur/colour-filtering-and-colour-pop-effects-using-opencv-python-3ce7d4576140
def saturate(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    #obtain the grayscale image of the original image
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    #set the bounds for the red hue
    lower_red = np.array([160,100,50])
    upper_red = np.array([180,255,255])

    #create a mask using the bounds set
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #create an inverse of the mask
    mask_inv = cv2.bitwise_not(mask)
    #Filter only the red colour from the original image using the mask(foreground)
    res = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    

def balancement(img_bgr):
    blue_adjustment = -10  # Adjust the blue channel
    green_adjustment = -5  # Adjust the green channel
    red_adjustment = 20  # Adjust the red channel

    # Apply color balance adjustments
    image = img_bgr.astype(np.float32)
    image[:, :, 2] = np.clip(image[:, :, 0] + blue_adjustment, 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] + green_adjustment, 0, 255)
    image[:, :, 0] = np.clip(image[:, :, 2] + red_adjustment, 0, 255)
    image = image.astype(np.uint8)

    return image
    
def filter_c(contours, image):
    height, width = image.shape[:2]

    image_area = width * height

    n_list = []

    for cont in contours:

        if len(cont) < 6:
            continue

        ellipse = cv2.fitEllipse(cont)
        c_area = cv2.contourArea(cont)

        # Extract the major and minor axes of the fitted ellipse
        major_axis, minor_axis = ellipse[1]

        # Calculate the aspect ratio (ratio of major axis to minor axis)
        ratio = major_axis / minor_axis

        if(c_area > 6**2 and c_area > image_area*.001 and 1/3 < ratio < 3):
            n_list.append(cont)
    return n_list

# bug: 35 41 96 98 104 131

for i in range(45, 172):
    fn = f"./img/IMG_0{i:03d}.png"
    img_bgr = cv2.imread(fn)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    edged = (edge_detection(blur(binary_three(img_bgr)))) 
    contours, hierachy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    _, (ax1) = plt.subplots(1,figsize=(20,20))
    plt.xticks([]),plt.yticks([])
    ax1.imshow(img_rgb);

    for cont in filter_c(contours, img_rgb):
        ax1.add_patch(matplotlib.patches.Polygon(cont[:, 0, :], edgecolor="green", linewidth=2, fill=False))
    
    plt.title(fn)
    plt.show()