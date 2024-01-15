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
from keras.models import load_model

shape_recognizer = load_model('./../../models/shape-recognizerv3-30eh.h5')

CATEGORIES = ['no turn', 'speed limit', 'access forbiden', 'no way', 'no parking', 'other']

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

        if(c_area > 6**2 and c_area > image_area*.0004 and 0.5 < ratio < 2):
            n_list.append(cont)
    return n_list

def pred_circle(img_bgr):
    # Load and preprocess the image
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = shape_recognizer.predict(img)

    # Interpret the prediction
    return prediction[0][0]

def get_min_x_max_x_min_y_max_y(contour):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')

    # Iterate through the points in the contour
    for point in contour:
        x, y = point[0]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return min_x, max_x, min_y, max_y

def predict_contour(contour):

    min_x, max_x, min_y, max_y = get_min_x_max_x_min_y_max_y(contour)

    width = max_x - min_x + 2*3
    height = max_y - min_y + 2*3

    offset_contour = contour.copy()
    for point in offset_contour:
        point[0][0] -= min_x - 3
        point[0][1] -= min_y - 3
        

    # Create a blank binary image
    new_image = np.zeros((height, width), dtype=np.uint8)


    cv2.drawContours(new_image, [offset_contour], 0, 255, thickness=2)

    pred = 100*(1 - pred_circle(new_image))

    return pred, new_image
    
def get_pannels(contours, threshold=80):

    res = []
    
    for contour in filter_c(contours, img_rgb):
        min_x, max_x, min_y, max_y = get_min_x_max_x_min_y_max_y(contour)
        pred, im = predict_contour(contour)
        if(pred < threshold):
            continue
        res.append(img(min_x, max_x, min_y, max_y, pred, im))
    
    if len(res) < 2:
        return res

    # remove the regions that are in common
    n_res = []

    for r in range(len(res)):
        included = False
        for k in range(r + 1, len(res)):
            if included or is_fully_included((res[r].min_x, res[r].max_x, res[r].min_y, res[r].max_y), (res[k].min_x, res[k].max_x, res[k].min_y, res[k].max_y)):
                included = True
        if not included:
            n_res.append(res[r])

    return n_res

class img:
    def __init__(self, min_x, max_x, min_y, max_y, pred, image):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.pred  = pred
        self.image = image



def is_fully_included(region1, region2):
    min_x1, max_x1, min_y1, max_y1 = region1
    min_x2, max_x2, min_y2, max_y2 = region2
    
    # Check if region1 is fully included in region2
    return (min_x1 >= min_x2) and (max_x1 <= max_x2) and (min_y1 >= min_y2) and (max_y1 <= max_y2)

def getim(pannels, im):
    res = []
    for i in pannels:
        res.append(im[i.min_y:i.max_y, i.min_x:i.max_x])
    return res

sign_model = load_model('./../../models/best_model-1.h5')

def predict_sign(photo):
    photo = cv2.resize(photo, (224, 224))
    photo = np.expand_dims(photo, axis=0)
    photo = photo / 255.0 # normalise the photo
    predictions = sign_model.predict(photo, verbose=0)
    return predictions

def predict_pannel_sign(pannels, background_image):
    for i in pannels:
        im = background_image[i.min_y : i.max_y, i.min_x : i.max_x]
        prediction = predict_sign(im)
        i.sign_prediction = prediction.argmax()

def disply_im(imgs, im):
    predict_pannel_sign(imgs, im)
    _, (ax1) = plt.subplots(1,figsize=(20,20))
    plt.xticks([]),plt.yticks([])
    ax1.imshow(im);
    for imm in imgs:
        ax1.add_patch(matplotlib.patches.Polygon([(imm.min_x, imm.min_y), (imm.max_x, imm.min_y), (imm.max_x, imm.max_y), (imm.min_x, imm.max_y)], edgecolor="green", linewidth=3, fill=False))
        plt.annotate(CATEGORIES[imm.sign_prediction], (imm.min_x + 2 , imm.min_y - 2), color='yellow', size=16)
    plt.show()

for i in range(100, 172):
    fn = f"./../../img/IMG_0{i:03d}.png"
    img_bgr = cv2.imread(fn)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    edged = edge_detection(blur(binary_three(img_bgr)))
    contours, hierachy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pannels = get_pannels(contours, 90)
    disply_im(pannels, img_rgb)