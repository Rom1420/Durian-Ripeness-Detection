import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from train_cnn import FruitMaturityCNN
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Define class mappings ===
CLASS_MAP = {
    "Ripe1": 0,  # No ripe
    "Ripe2": 1,  # Mature
    "Ripe3": 2,  # Ripe
    "Ripe4": 3,  # Overripe
}
CLASS_NAMES = ["No ripe", "Mature", "Ripe", "Overripe"]

# === 2. Define image preprocessing steps ===
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# === 3. Load a trained CNN model from disk ===
def load_cnn_model(path):
    model = FruitMaturityCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# === 4. Define model paths and dataset settings ===
model_paths = {
    "original": "model/cnn_original.pth",
    "crop": "model/cnn_crop.pth",
    "mixed": "model/cnn_mixed.pth",
}

model_names = {
    "original": "CNN trained on Original",
    "crop": "CNN trained on Crop",
    "mixed": "CNN trained on Mixed",
}

dataset_types = ["original", "crop", "mixed"]
base_path = "dbsets"

# === 5. Initialize result containers ===
results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
per_class_accuracy = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
)

# === 6. Evaluate all models on all datasets ===
for model_key, model_path in model_paths.items():
    model = load_cnn_model(model_path)
    print(f"\n🚀 Evaluating model {model_key} ({model_names[model_key]})")

    for dataset_type in dataset_types:
        dataset_path = os.path.join(base_path, dataset_type)

        for class_folder in os.listdir(dataset_path):
            if class_folder not in CLASS_MAP:
                continue
            label_idx = CLASS_MAP[class_folder]
            class_path = os.path.join(dataset_path, class_folder)

            for fname in tqdm(
                os.listdir(class_path),
                desc=f"{model_key} on {dataset_type}/{class_folder}",
            ):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(class_path, fname)
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    pred = torch.argmax(logits).item()

                results[model_key][dataset_type]["total"] += 1
                per_class_accuracy[model_key][dataset_type][label_idx]["total"] += 1

                if pred == label_idx:
                    results[model_key][dataset_type]["correct"] += 1
                    per_class_accuracy[model_key][dataset_type][label_idx][
                        "correct"
                    ] += 1

# === 7. Print global accuracy summary ===
print("\n📊 Summary table (accuracy in %):\n")
header = f"{'Model':<25}" + "".join([f"{ds:^12}" for ds in dataset_types]) + "Average"
print(header)
print("-" * (25 + 12 * len(dataset_types) + 10))

summary_data = {}

for model_key in model_paths:
    row = f"{model_names[model_key]:<25}"
    total_acc = 0
    count = 0
    summary_data[model_key] = []
    for ds in dataset_types:
        correct = results[model_key][ds]["correct"]
        total = results[model_key][ds]["total"]
        acc = 100 * correct / total if total > 0 else 0.0
        summary_data[model_key].append(acc)
        row += f"{acc:>10.2f}%  "
        total_acc += acc
        count += 1
    avg = total_acc / count
    row += f"{avg:>8.2f}%"
    print(row)

# === 8. Plot per-class accuracy heatmaps ===
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
for i, model_key in enumerate(model_paths):
    heatmap_data = []
    for cls_idx in range(4):
        row = []
        for ds in dataset_types:
            total = per_class_accuracy[model_key][ds][cls_idx]["total"]
            correct = per_class_accuracy[model_key][ds][cls_idx]["correct"]
            acc = 100 * correct / total if total > 0 else 0.0
            row.append(acc)
        heatmap_data.append(row)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=dataset_types,
        yticklabels=CLASS_NAMES,
        ax=axes[i],
    )
    axes[i].set_title(f"Class-wise Accuracy – {model_names[model_key]}")
    axes[i].set_xlabel("Dataset")
    axes[i].set_ylabel("Class")

plt.tight_layout()
plt.savefig("per_class_accuracy_heatmaps.png")
plt.close()

"✅ Plots saved as 'per_class_accuracy_heatmaps.png'."
