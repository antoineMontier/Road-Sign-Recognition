from PIL import Image
from augly import perturb
import os

# Function to apply AugLy transformations to an image
def apply_augly_transformations(image):
    return perturb.apply_one(
        image,
        perturb.ColorJitter(saturation=1.5, contrast=1.5, brightness=0.5),
        perturb.GaussianBlur(sigma=1.0),
        perturb.Resample(scale=0.5),
    )

if __name__ == "__main__":
    input_folder = "artifical_data"  # Change this to the folder containing your images
    output_folder = "augmented_artificial_data"

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = Image.open(input_path)

            # Apply AugLy transformations
            augmented_image = apply_augly_transformations(image)

            # Save the augmented image
            augmented_image.save(output_path)

    print(f"Augmented images are saved in the '{output_folder}' folder.")
