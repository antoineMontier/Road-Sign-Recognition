import math
import random
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

def ellipse_gen(self, width_shape, height_shape, rotate_deg):
    """Draw an ellipse with noise"""
    a = float(width_shape) / 2.0
    b = float(height_shape) / 2.0
    h = float(self.size_cnv["width"]) / 2.0
    k = float(self.size_cnv["height"]) / 2.0

    # Calculate the rotation matrix for the ellipse
    rotation_matrix = [
        [math.cos(math.radians(rotate_deg)), -math.sin(math.radians(rotate_deg))],
        [math.sin(math.radians(rotate_deg)), math.cos(math.radians(rotate_deg))]]

    x_off = random(100)
    stack_vertex = []
    for x_obs in range(floor(h - a) + 1, floor(h + a) + 1):
        t = (x_obs - h) / a
        y = k + b * t

        # Apply rotation to get the final coordinates
        rotated_x = h + t * a * rotation_matrix[0][0] + y * b * rotation_matrix[0][1]
        rotated_y = k + t * a * rotation_matrix[1][0] + y * b * rotation_matrix[1][1]

        noise_1d = map(noise(x_off), 0, 1, -margin_off_noise, margin_off_noise)
        stack_vertex.append((rotated_x, rotated_y + noise_1d))
        stack_vertex.insert(0, (rotated_x, rotated_y - noise_1d))
        x_off += self.inc_time_noise

    line_color = self.list_color[int(random(len(self.list_color)))]
    pushMatrix()
    self.rotate_epi(rotate_deg)
    stroke(line_color[0], line_color[1], line_color[2])
    strokeWeight(self.stroke_weight)
    noFill()
    beginShape()
    for coord in stack_vertex:
        vertex(coord[0], coord[1])
    endShape()
    popMatrix()

def circle_gen(self, height_shape, width_shape, rotate_deg):
    """Return ellipse almost like circle with noise"""
    # Ref: https://saylordotorg.github.io/text_intermediate-algebra/s11-03-ellipses.html
    a = float(width_shape) / 2.0
    b = float(height_shape) / 2.0
    c = sqrt(a ** 2 + b ** 2)
    h = float(self.size_cnv["width"]) / 2.0
    k = float(self.size_cnv["height"]) / 2.0
    l = sqrt(h ** 2 + k ** 2)
    x_off = random(100)
    stack_vertex = []
    buff_noise = map(c, 0, l, 1, 2.5)
    margin_off_noise = self.stroke_weight * self.ratio_margin_noise * buff_noise
    for x_obs in range(floor(h - a) + 1, floor(h + a) + 1):
        s = sqrt(1 - (float(x_obs - h) / a) ** 2) * b
        noise_1d = map(noise(x_off), 0, 1, -margin_off_noise, margin_off_noise)
        stack_vertex.append((x_obs, floor(k + s + noise_1d)))
        stack_vertex.insert(0, (x_obs, floor(k - s + noise_1d)))
        x_off += self.inc_time_noise
    line_color = self.list_color[int(random(len(self.list_color)))]
    pushMatrix()
    self.rotate_epi(rotate_deg)
    stroke(line_color[0], line_color[1], line_color[2])
    strokeWeight(self.stroke_weight)
    noFill()
    beginShape()
    for coord in stack_vertex:
        vertex(coord[0], coord[1])
    endShape()
    popMatrix()

def setup():
    shape_.img_mask_dashed_noise = loadImage("assets/dashed_line_noise.png")
    size(size_canvas["width"], size_canvas["height"])

constants = {"PATH_DEST": os.getcwd(),
             "NUM_TRAIN": 1500,
             "NUM_VAL": 0,
             "NUM_TEST": 0}

# Inject value by env variables
for key_constant in constants:
    try:
        if os.environ[key_constant]:
            # Validate env variable
            type_constant = key_constant.split("_")[0]
            if type_constant == "NUM":
                constants[key_constant] = abs(int(os.environ[key_constant]))
            elif type_constant == "PATH" and os.path.exists(os.environ[key_constant]):
                constants[key_constant] = os.environ[key_constant]
            else:
                print("{} is invalid".format(key_constant))
    except KeyError:
        print("{} enviroment variable is not set".format(key_constant))

base_dir = os.path.join(constants["PATH_DEST"], "dataset")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")


img_counter = 0
# Order: start => train => val => test => stop
toggle_bool = True
base_class_dir = train_dir
print("Please wait, currently creating a dataset...")

def draw():
    global base_class_dir
    global img_counter
    global toggle_bool
    if img_counter == constants["NUM_TRAIN"]:
        base_class_dir = val_dir
    if img_counter == constants["NUM_TRAIN"] + constants["NUM_VAL"]:
        base_class_dir = test_dir
    shape_.stroke_weight = int(random(4, 12))
    # shape_.inc_time_noise = random(0.025, 0.035)
    # Rectangle
    background(255)
    min_length = int(random(50, 150))
    max_length = int(random(min_length + 30, 200))
    shape_.rectangle_gen(height_shape=max_length if toggle_bool else min_length,
                         width_shape=min_length if toggle_bool else max_length,
                         rotate_deg=int(random(-5, 5)))
    shape_.mask_dashed_line(activate=bool(img_counter % 4 == 1 or img_counter % 4 == 2),
                            counter_deg=img_counter)
    save(os.path.join(base_class_dir, "rectangle", "rectangle-{}.jpg".format(img_counter)))
    # Circle
    background(255)
    max_length = int(random(min_length, min_length + 50))
    shape_.circle_gen(height_shape=max_length if toggle_bool else min_length,
                      width_shape=min_length if toggle_bool else max_length,
                      rotate_deg=int(random(-180, 180)))
    shape_.mask_dashed_line(activate=toggle_bool, counter_deg=img_counter)
    save(os.path.join(base_class_dir, "circle", "circle-{}.jpg".format(img_counter)))
    # Ellipse
    background(255)
    max_length = int(random(min_length, min_length + 50))
    shape_.ellipse_gen(height_shape=max_length if toggle_bool else min_length,
                      width_shape=min_length if toggle_bool else max_length,
                      rotate_deg=int(random(-180, 180)))
    shape_.mask_dashed_line(activate=toggle_bool, counter_deg=img_counter)
    save(os.path.join(base_class_dir, "circle", "circle-{}.jpg".format(img_counter)))
    # # Kite
    # background(255)
    # max_length = int(random(min_length + 30, 200))
    # shape_.kite_gen(height_shape=max_length,
    #                 width_shape=min_length,
    #                 rotate_deg=int(random(-15, 15)),
    #                 flip_vert=toggle_bool)
    # shape_.mask_dashed_line(activate=bool(img_counter % 4 == 1 or img_counter % 4 == 2),
    #                         counter_deg=img_counter)
    # save(os.path.join(base_class_dir, "kite", "kite-{}.jpg".format(img_counter)))
    # # Rhombus
    # background(255)
    # shape_.rhombus_gen(height_shape=max_length if toggle_bool else min_length,
    #                    width_shape=min_length if toggle_bool else max_length,
    #                    rotate_deg=int(random(-5, 5)))
    # shape_.mask_dashed_line(activate=bool(img_counter % 4 == 1 or img_counter % 4 == 2),
    #                         counter_deg=img_counter)
    # save(os.path.join(base_class_dir, "rhombus", "rhombus-{}.jpg".format(img_counter)))
    # # Parallelogram
    # background(255)
    # min_length = int(random(50, 200))
    # max_length = int(random(min_length, 200))
    # shape_.parallelogram_gen(height_shape=min_length,
    #                          width_shape=max_length,
    #                          rotate_deg=int(random(-10, 10)),
    #                          ratio_base=random(0.4, 0.7),
    #                          flip_horz=toggle_bool)
    # shape_.mask_dashed_line(activate=bool(img_counter % 4 == 1 or img_counter % 4 == 2),
    #                         counter_deg=img_counter)
    # save(os.path.join(base_class_dir, "parallelogram", "parallelogram-{}.jpg".format(img_counter)))
    # # Square
    # background(255)
    # shape_.square_gen(length_shape=int(random(50, 200)),
    #                   rotate_deg=int(random(-5, 5)))
    # shape_.mask_dashed_line(activate=toggle_bool, counter_deg=img_counter)
    # save(os.path.join(base_class_dir, "square", "square-{}.jpg".format(img_counter)))
    # # Trapezoid
    # background(255)
    # shape_.trapezoid_gen(height_shape=min_length,
    #                      width_shape=max_length,
    #                      rotate_deg=int(random(-10, 10)),
    #                      ratio_parallel=random(0.35, 0.55),
    #                      flip_vert=toggle_bool)
    # shape_.mask_dashed_line(activate=bool(img_counter % 4 == 1 or img_counter % 4 == 2),
    #                         counter_deg=img_counter)
    # save(os.path.join(base_class_dir, "trapezoid", "trapezoid-{}.jpg".format(img_counter)))
    # # Triangle
    # background(255)
    # min_length = int(random(50, 180))
    # max_length = int(random(min_length, 200))
    # shape_.triangle_gen(height_shape=max_length if toggle_bool else min_length,
    #                     width_shape=min_length if toggle_bool else max_length,
    #                     rotate_deg=int(random(-15, 15)),
    #                     flip_vert=toggle_bool)
    # shape_.mask_dashed_line(activate=bool(img_counter % 4 == 1 or img_counter % 4 == 2),
    #                         counter_deg=img_counter)
    # save(os.path.join(base_class_dir, "triangle", "triangle-{}.jpg".format(img_counter)))
    if img_counter == constants["NUM_TRAIN"] + constants["NUM_VAL"] + constants["NUM_TEST"] - 1:
        background(200)
        msg = "Dataset Created!"
        print(msg)
        pushStyle()
        fill(0)
        textSize(size_canvas["width"] * 0.1)
        textAlign(CENTER, CENTER)
        text(msg, size_canvas["width"] / 2.0, size_canvas["height"] / 2.0)
        popStyle()
        noLoop()
    img_counter += 1
    toggle_bool = not toggle_bool

draw()