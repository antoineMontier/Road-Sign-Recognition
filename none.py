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


for i in range(1, 172):
    fn = f"./img/IMG_0{i:03d}.png"
    img = cv2.imread(fn)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 40, 40])
    upper_red = np.array([12, 255, 255])

    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    red_segmented = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.cvtColor(red_segmented,cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    plt.imshow(img)
    plt.show()

    continue

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 40, 40])
    upper_red = np.array([12, 255, 255])

    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    red_segmented = cv2.bitwise_and(img, img, mask=mask)

    plt.imshow(red_segmented)
    plt.title(fn)
    plt.show()