import os
import cv2
from ultralytics import YOLO

# === CONFIGURATION ===

# Path to the YOLOv8 model (for detection)
yolo_model_path = "model/yolov8m.pt"

# Base folder containing the original images
base_dir = "database"

# Folder where cropped images will be saved
crop_dir = os.path.join(base_dir, "crop")
os.makedirs(crop_dir, exist_ok=True)  # Create the crop directory if it doesn't exist

# Load the YOLO model
model = YOLO(yolo_model_path)

# Get all image filenames in the base directory (jpg, jpeg, png)
images = [
    f for f in os.listdir(base_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Process each image one by one
for img_name in images:
    img_path = os.path.join(base_dir, img_name)  # Full path to the image
    img = cv2.imread(img_path)  # Load image with OpenCV

    results = model(img_path)  # Run YOLOv8 inference
    boxes = (
        results[0].boxes.xyxy.cpu().numpy()
    )  # Get bounding box coordinates (x1, y1, x2, y2)

    if len(boxes) == 0:
        # If no object is detected, skip this image
        print(f"[!] No fruit detected in {img_name}, skipped.")
        continue

    # Take the first detected bounding box
    x1, y1, x2, y2 = map(int, boxes[0])
    crop = img[y1:y2, x1:x2]  # Crop the image using the bounding box

    # Construct a new filename for the cropped image
    new_name = os.path.splitext(img_name)[0] + "_crop.jpg"
    output_path = os.path.join(crop_dir, new_name)

    # Save the cropped image
    cv2.imwrite(output_path, crop)
    print(f"✅ Saved: {output_path}")

# Final confirmation message
print("\n✅ All cropping operations completed.")
