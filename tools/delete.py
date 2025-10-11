#----- Deletes images from GTSDB that do not contain any traffic sign -----#

import os

txt_dir = "./data/labels/yolo"
img_dir = "./data/images/yolo"

# all .txt files
txt_basenames = {os.path.splitext(f)[0] for f in os.listdir(txt_dir) if f.endswith(".txt")}

# go through all image files
for img_file in os.listdir(img_dir):
    if img_file.endswith(".jpg"):
        base = os.path.splitext(img_file)[0] 
        
        # check if associated .txt exists
        if base not in txt_basenames:
            img_path = os.path.join(img_dir, img_file)
            os.remove(img_path)
            print(f"🗑️ deleted: {img_file}")

