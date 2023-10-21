from PIL import Image, ImageDraw
import math
import random

def add_imperfections(shape, width, height):
    # Add imperfections by introducing random variations
    max_offset = min(width, height) // 10  # Adjust the magnitude of imperfections as needed

    # Randomly adjust the position of the shape
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    shape = shape.transform((width, height), Image.AFFINE, (1, 0, offset_x, 0, 1, offset_y))

    # Randomly adjust the size of the shape
    size_variation = random.uniform(0.9, 1.1)  # Adjust the range as needed
    shape = shape.resize((int(width * size_variation), int(height * size_variation)))

    # Randomly adjust the outline color
    outline_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw = ImageDraw.Draw(shape)
    draw.rectangle([0, 0, width, height], outline=outline_color)

    return shape

def create_random_image(width, height):
    # Create a new image with a white background
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Randomly select the type of shape (circle, ellipse, or half-circle)
    shape_type = random.choice(["circle", "ellipse", "half_circle"])

    print(shape_type)

    if shape_type == "circle":
        # Calculate the maximum radius that fits within the image dimensions
        max_radius = min(width, height) // 2
        radius = random.randint(10, max_radius)
        
        # Calculate the center point randomly
        center = (width // 2, height // 2)

        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline="black")

    elif shape_type == "ellipse":
        # Calculate the dimensions for the ellipse
        min_axis = min(width, height) // 4
        major_axis = random.randint(min_axis, min(width, height) // 2)
        minor_axis = random.randint(min_axis, major_axis)
        
        left = (width - major_axis) // 2
        top = (height - minor_axis) // 2

        right = left + major_axis
        bottom = top + minor_axis
        

        # Calculate the bottom-right coordinates
        right = left + major_axis
        bottom = top + minor_axis


        draw.ellipse((left, top, right, bottom), outline="black")

    else:
        # Calculate the radius for the half-circle
        radius = min(width, height) // 2
        
        # Calculate the center point randomly
        center = (width // 2, height // 2)
        start_angle = random.randint(0, 180)
        end_angle = start_angle + 180
        # Randomly choose between a visible or invisible diameter (line)
        if random.choice([True, False]):
            
            # Calculate the starting and ending angles for the half-circle
            start_angle = random.randint(0, 180)
            end_angle = start_angle + 180
            print("\ttrue")
            draw.pieslice((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), start_angle, end_angle, outline="black")
        else:
            print("\tfalse")
            # Draw a filled half-circle
            draw.arc((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), 0, 180, fill=0)

    return image

if __name__ == "__main__":
    width, height = 224, 224
    num_images = 10  # Change this to the number of images you need

    for i in range(num_images):
        image = create_random_image(width, height)
        image = add_imperfections(image, width, height)
        image.save(f"artificial_data/IMG_{i:04d}.png")

    print(f"Generated {num_images} random images for machine learning.")
