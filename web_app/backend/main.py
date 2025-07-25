import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
import torch
import torch.nn.functional as F
from train_cnn import FruitMaturityCNN

# === FastAPI initialization ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Class labels ===
classes = ["No ripe", "Mature", "Ripe", "Overripe"]

# === Image preprocessing for CNN ===
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# === Load trained CNN model ===
def load_cnn_model(path):
    model = FruitMaturityCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


cnn_model = load_cnn_model("model/cnn_original.pth")

# === Load font for annotations ===
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()


# === Prediction endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    contents = await file.read()
    filename = f"{uuid.uuid4().hex}.jpg"
    upload_path = os.path.join("static", "uploads", filename)
    os.makedirs(os.path.dirname(upload_path), exist_ok=True)
    with open(upload_path, "wb") as f:
        f.write(contents)

    # Open image
    image = Image.open(upload_path).convert("RGB")

    # Prepare input for CNN
    input_tensor = transform(image).unsqueeze(0)

    # Predict with CNN
    with torch.no_grad():
        logits = cnn_model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    label_text = f"{classes[pred_idx]} ({confidence * 100:.1f}%)"

    # Annotate image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    draw.text((10, 10), label_text, fill="green", font=font)

    # Save annotated result
    result_filename = f"cnn_result_{os.path.splitext(filename)[0]}.jpg"
    result_path = os.path.join("static", "results", result_filename)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    annotated.save(result_path)

    return {
        "cnn_prediction": {
            "class": classes[pred_idx],
            "confidence": f"{confidence * 100:.1f}%",
        },
        "annotated_image": f"/static/results/{result_filename}",
    }