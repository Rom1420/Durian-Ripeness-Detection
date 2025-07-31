# detectron2_durian_training.py
# -------------------------------------------------
# Train a Mask R-CNN model on a custom durian dataset using COCO format.
# After training, runs inference on the validation set and saves:
# - Annotated segmentation images
# - Contour coordinates and confidence scores
# -------------------------------------------------

import os
import shutil
import cv2
import numpy as np
import torch

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# === 1. Register COCO-format datasets ===
register_coco_instances(
    "durian_train",
    {},
    "/home/luca/Documents/Stage/programme/dataset_cnn/annotations/instances_train.json",
    "/home/luca/Documents/Stage/programme/dataset_cnn/train",
)
register_coco_instances(
    "durian_val",
    {},
    "/home/luca/Documents/Stage/programme/dataset_cnn/annotations/instances_val.json",
    "/home/luca/Documents/Stage/programme/dataset_cnn/val",
)

# === 2. Configure the model ===
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("durian_train",)
cfg.DATASETS.TEST = ("durian_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.OUTPUT_DIR = "./output_durian"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.DEVICE = "cpu"  # Use "cuda" if a GPU is available

# Delete old metrics file if it exists
metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
if os.path.exists(metrics_path):
    os.remove(metrics_path)
    print("✅ Old metrics.json deleted")

# === 3. Train the model ===
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# === 4. Load the trained model for inference ===
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for showing predictions
predictor = DefaultPredictor(cfg)

# === 5. Prepare output directories ===
segmentation_dir = os.path.join(cfg.OUTPUT_DIR, "segmentation_cnn")
coords_dir = os.path.join(cfg.OUTPUT_DIR, "coordonnees_cnn")
os.makedirs(segmentation_dir, exist_ok=True)
os.makedirs(coords_dir, exist_ok=True)


# Function to clean an output directory
def clear_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)


clear_folder(segmentation_dir)
clear_folder(coords_dir)

# === 6. Run inference on the validation images and save contours ===
val_dir = "/home/luca/Documents/Stage/programme/dataset_cnn/val"
val_images = [f for f in os.listdir(val_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

for image_name in val_images:
    image_path = os.path.join(val_dir, image_name)
    image = cv2.imread(image_path)
    outputs = predictor(image)

    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    for i, mask in enumerate(masks):
        if classes[i] == 0:  # Class 0 = durian
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                contour_coords = max(contours, key=cv2.contourArea).squeeze().tolist()
                contour_np = np.array(contour_coords, dtype=np.int32).reshape(-1, 1, 2)
                cv2.drawContours(image, [contour_np], -1, (0, 255, 0), 2)

                confidence = scores[i]
                label = f"Durian: {confidence:.2f}"
                cv2.putText(
                    image,
                    label,
                    (contour_np[0][0][0], contour_np[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                # Save results
                output_img_path = os.path.join(
                    segmentation_dir, f"{os.path.splitext(image_name)[0]}_contour.png"
                )
                output_txt_path = os.path.join(
                    coords_dir, f"contours_{os.path.splitext(image_name)[0]}.txt"
                )
                cv2.imwrite(output_img_path, image)
                with open(output_txt_path, "w") as f:
                    f.write(f"Contour: {contour_coords}\n")
                    f.write(f"Confidence: {confidence}\n")

                print(f"✅ {image_name} → Contour saved (confidence: {confidence:.2f})")
