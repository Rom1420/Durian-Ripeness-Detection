from ultralytics import YOLO
import os

# Load the YOLOv8 segmentation model
model = YOLO("model/yolon8m-seg.pt")

# Path to the image for prediction
image_path = r"image.png"

# Output folder settings
output_folder = "runs/segment/predict"
subfolder_name = "predict1"
subfolder_path = os.path.join(output_folder, subfolder_name)

# Run prediction and save results (images, labels, etc.)
model.predict(
    source=image_path,
    conf=0.81,
    save=True,
    save_txt=True,
    save_crop=False,
    project=output_folder,
    name=subfolder_name,
    exist_ok=True,
)

print(f"✅ Prediction completed for: {image_path}")
print(f"📂 Results saved in: {subfolder_path}/")

# Run a second prediction with a lower confidence threshold to extract masks
results = model.predict(source=image_path, conf=0.1)
masks = results[0].masks  # List of masks (if available)
results[0].show()  # Display the prediction in a pop-up window

# Display number of masks detected (if any)
if masks is not None:
    print(f"Number of masks detected: {len(masks.data)}")
else:
    print("❌ No masks detected.")
