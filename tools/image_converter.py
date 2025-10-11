#----- Converts an .ppm file to .jpg -----#

from PIL import Image # type: ignore
import os

INPUT_FOLDER = "C:/Users/dawid/Downloads/GTSRB/Final_Test"
OUTPUT_FOLDER = "./data/images/yolo_only"

# Go through all files in given INPUT_FOLDER
for filename in os.listdir(INPUT_FOLDER):

    # Check if file extension is .ppm
    if filename.lower().endswith(".ppm"):
        ppm_path = os.path.join(INPUT_FOLDER, filename) # .ppm-file path
        jpg_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + ".jpg") # .jpg-file path
        
        # Open image, convert to RGB and save
        with Image.open(ppm_path) as img:
            rgb_img = img.convert("RGB") 
            rgb_img.save(jpg_path)
            
        print(f"converted {filename} to {jpg_path}")
