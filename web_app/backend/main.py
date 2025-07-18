import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from ultralytics import YOLO
from train_cnn import FruitMaturityCNN
from fastapi.middleware.cors import CORSMiddleware

# === Initialisation FastAPI ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Classes ===
classes = ["No ripe", "Mature", "Ripe", "Overripe"]

# === Prétraitement CNN ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Chargement du modèle CNN Original ===
def load_cnn_model(path):
    model = FruitMaturityCNN()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

cnn_model = load_cnn_model("model/cnn_original.pth")

# === Chargement du modèle YOLO ===
yolo_model = YOLO("model/yolov8m.pt")

# === Font ===
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

# === Endpoint de prédiction ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join("static", "uploads", filename)
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(contents)

    # Lecture et conversion
    img_bgr = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # === Prédiction sur image originale ===
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = cnn_model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        cnn_pred = {
            "class": classes[pred_idx],
            "confidence": f"{confidence*100:.1f}%"
        }

    # === Détection avec YOLO ===
    results = yolo_model(input_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return JSONResponse({"error": "Aucun fruit détecté", "cnn_prediction": cnn_pred})

    output_images = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img_rgb[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop)
        input_tensor = transform(pil_crop).unsqueeze(0)

        with torch.no_grad():
            logits = cnn_model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze()
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            label = f"{classes[pred_idx]} ({confidence * 100:.1f}%)"

        crop_annotated = Image.fromarray(cv2.cvtColor(crop.copy(), cv2.COLOR_RGB2BGR))
        draw = ImageDraw.Draw(crop_annotated)
        draw.text((10, 10), label, fill=(0, 255, 0), font=font)

        save_name = f"compare_{os.path.splitext(filename)[0]}_fruit{i+1}.png"
        save_path = os.path.join("static", "results", save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        crop_annotated.save(save_path)
        output_images.append(f"/static/results/{save_name}")
        
    img_draw = Image.fromarray(cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR))
    draw = ImageDraw.Draw(img_draw)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = os.path.basename(output_images[i]).split("_")[-1].replace(".png", "")
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"Fruit {i+1}", fill="red", font=font)

    # Sauvegarder image annotée YOLO
    segmented_filename = f"yolo_segmented_{os.path.splitext(filename)[0]}.jpg"
    segmented_path = os.path.join("static", "results", segmented_filename)
    img_draw.save(segmented_path)


    return {
        "yolo_results": output_images,
        "cnn_prediction_on_original": cnn_pred,
        "yolo_segmented_image": f"/static/results/{segmented_filename}"
    }
