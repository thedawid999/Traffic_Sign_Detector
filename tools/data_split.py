#----- used to split data equally across all classes in Training and Validation Sets -----#

import os
import shutil
from sklearn.model_selection import train_test_split # type: ignore

label_folder = "./data/labels/yolo/"       
image_folder = "./data/images/yolo/"     
target_folder = "./data/yolo/"       
image_ext = ".jpg"                    
label_ext = ".txt"                 
val_size = 0.2  # 20% Validation

# get all folder directories
train_image_folder = os.path.join(target_folder, "train/images")
train_label_folder = os.path.join(target_folder, "train/labels")
val_image_folder = os.path.join(target_folder, "val/images")
val_label_folder = os.path.join(target_folder, "val/labels")

for folder in [train_image_folder, train_label_folder, val_image_folder, val_label_folder]:
    os.makedirs(folder, exist_ok=True)

# go through all label files
labels = [f for f in os.listdir(label_folder) if f.endswith(label_ext)]
labels.sort()

# split data in Training and Validation
train_labels, val_labels = train_test_split(labels, test_size=val_size, random_state=42)


# copies all splitted files into destination folder
def copy_files(label_list, dest_image_folder, dest_label_folder):
    for label_file in label_list:
        base_name = os.path.splitext(label_file)[0]
        image_file = base_name + image_ext

        # source and destination
        src_label = os.path.join(label_folder, label_file)
        src_image = os.path.join(image_folder, image_file)
        dst_label = os.path.join(dest_label_folder, label_file)
        dst_image = os.path.join(dest_image_folder, image_file)

        # copy files
        if os.path.exists(src_image):
            shutil.copy2(src_label, dst_label)
            shutil.copy2(src_image, dst_image)
        else:
            print(f"Image {src_image} does not exist!")

# copy files
copy_files(train_labels, train_image_folder, train_label_folder)
copy_files(val_labels, val_image_folder, val_label_folder)
