import os
import shutil
import random

# Paths
img_dir = "dataset/images"
label_dir = "dataset/labels/train"
train_img_dir = "dataset/images/train"
val_img_dir = "dataset/images/val"
train_label_dir = "dataset/labels/train"
val_label_dir = "dataset/labels/val"

# Create required directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get the list of image files to split
images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
random.shuffle(images)
split_index = int(len(images) * 0.8)
train_images = images[:split_index]
val_images = images[split_index:]


# Function to move both image and corresponding label
def move_files(image_list, target_img_dir, target_label_dir):
    for img_name in image_list:
        base = os.path.splitext(img_name)[0]
        label_name = base + ".txt"

        # Move image
        shutil.move(
            os.path.join(img_dir, img_name), os.path.join(target_img_dir, img_name)
        )

        # Move label if it exists
        label_src = os.path.join(label_dir, label_name)
        label_dst = os.path.join(target_label_dir, label_name)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)


# Apply the file splitting and moving
move_files(train_images, train_img_dir, train_label_dir)
move_files(val_images, val_img_dir, val_label_dir)

print("✅ Dataset reorganized for YOLOv8")
