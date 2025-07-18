import os
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from ultralytics import YOLO
from train_cnn import FruitMaturityCNN
from PIL import Image, ImageDraw, ImageFont

# === 1. Target classes ===
classes = ["No ripe", "Mature", "Ripe", "Overripe"]

# === 2. Image preprocessing pipeline ===
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Resize crop to match CNN input
        transforms.ToTensor(),  # Convert PIL to tensor
        transforms.Normalize([0.5], [0.5]),  # Normalize pixel values to [-1, 1]
    ]
)


# === 3. Load the trained CNN models ===
def load_cnn_model(path):
    model = FruitMaturityCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))  # Load weights on CPU
    model.eval()  # Set model to evaluation mode
    return model


# Dictionary containing the three CNN models to compare
models = {
    "Original": load_cnn_model("model/cnn_original.pth"),
    "Crop": load_cnn_model("model/cnn_crop.pth"),
    "Mixed": load_cnn_model("model/cnn_mixed.pth"),
}

# === 4. Load YOLO model for object detection ===
yolo_model = YOLO("model/yolov8m.pt")

# === 5. Input and output directories ===
test_dir = "test"
output_dir = "test/compare_results"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# === 6. Font for annotation ===
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()  # Fallback if Arial is not available

# === 7. Loop over all test images ===
for filename in os.listdir(test_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # Skip non-image files

    img_path = os.path.join(test_dir, filename)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(
        img_bgr, cv2.COLOR_BGR2RGB
    )  # Convert to RGB for PIL compatibility

    # Run YOLO object detection
    results = yolo_model(img_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding box coordinates

    if len(boxes) == 0:
        print(f"[!] No fruits detected in {filename}")
        continue

    # For each detected fruit
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img_rgb[y1:y2, x1:x2]  # Crop the region of the detected fruit
        pil_crop = Image.fromarray(crop)  # Convert to PIL for preprocessing
        input_tensor = transform(pil_crop).unsqueeze(0)  # Add batch dimension

        preds_img = []  # List to store prediction images for side-by-side comparison

        # Run prediction for each CNN model
        for model_name, model in models.items():
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze()  # Get class probabilities
                pred_idx = torch.argmax(probs).item()  # Predicted class index
                confidence = probs[pred_idx].item()  # Confidence score
                label = f"{classes[pred_idx]} ({confidence * 100:.1f}%)"

            # Annotate the crop with model name and prediction
            result_img = Image.fromarray(cv2.cvtColor(crop.copy(), cv2.COLOR_RGB2BGR))
            draw = ImageDraw.Draw(result_img)
            draw.text((10, 10), f"{model_name}\n{label}", fill=(0, 255, 0), font=font)
            preds_img.append(result_img)

        # Combine all annotated images horizontally
        widths, heights = zip(*(img.size for img in preds_img))
        total_width = sum(widths)
        max_height = max(heights)
        combined = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for img in preds_img:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save the comparison image
        save_name = f"compare_{os.path.splitext(filename)[0]}_fruit{i+1}.png"
        save_path = os.path.join(output_dir, save_name)
        combined.save(save_path)

print("✅ Comparison images saved in test/compare_results/")
