import cv2
import os

# Define the directories
directories = ['A/', 'B/', 'C/', 'D/', 'E/', 'F/']
input_path = "./../../Training/augmentation1/"
output_path = "./../../Training/augmentation3/"

# Make sure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Define the zoom factor
ZOOM_FACTOR_1 = 1.33
ZOOM_FACTOR_2 = 1.66


# Function to zoom an image
def zoom_image(img, factor):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    width_scaled, height_scaled = int(img.shape[1] / factor), int(img.shape[0] / factor)
    left_x, right_x = center_x - width_scaled // 2, center_x + width_scaled // 2
    top_y, bottom_y = center_y - height_scaled // 2, center_y + height_scaled // 2

    # Crop and resize to simulate zoom
    img_cropped = img[top_y:bottom_y, left_x:right_x]
    img_zoomed = cv2.resize(img_cropped, (img.shape[1], img.shape[0]))
    return img_zoomed

# Process the images in each directory
for directory in directories:
    full_input_path = os.path.join(input_path, directory)
    full_output_path = os.path.join(output_path, directory)
    os.makedirs(full_output_path, exist_ok=True)  # Create the output subdirectory
    
    # Iterate through each image in the directory
    for filename in os.listdir(full_input_path):
        image_path = os.path.join(full_input_path, filename)
        image = cv2.imread(image_path)

        # Check if the image is already the correct size
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # Save the original image
        cv2.imwrite(os.path.join(full_output_path, filename), image)
        
        # Create and save the zoomed image
        image_zoomed = zoom_image(image, ZOOM_FACTOR_1)

        cv2.imwrite(os.path.join(full_output_path, 'z' + filename), image_zoomed)

        image_dzoomed = zoom_image(image, ZOOM_FACTOR_2)

        cv2.imwrite(os.path.join(full_output_path, 'zz' + filename), image_dzoomed)

print("Processing complete.")
