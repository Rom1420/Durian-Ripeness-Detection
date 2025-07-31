# Detectron2 Inference Script for Durian Segmentation

import os
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import shutil

# Configure the model
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = "path/to/model.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if GPU is available

predictor = DefaultPredictor(cfg)

# Directories
real_images_dir = "path/to/dataset"
output_dir = "path/to/output/dir"

coords_dir = os.path.join(output_dir, "coords_masks")
annotated_dir = os.path.join(output_dir, "annotated_images")

# Reset output directories
for d in [coords_dir, annotated_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# Inference loop
for img_name in os.listdir(real_images_dir):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(real_images_dir, img_name)
    image = cv2.imread(img_path)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    if not hasattr(instances, "pred_masks"):
        print(f"No masks predicted for {img_name}")
        continue

    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    best_score = 0
    best_mask_coords = None
    best_mask = None

    # Find the best durian mask (class 0)
    for i, mask in enumerate(masks):
        if classes[i] == 0:  # durian class
            confidence = scores[i]
            if confidence > best_score:
                contours, _ = cv2.findContours(
                    (mask * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    contour_coords = (
                        max(contours, key=cv2.contourArea).squeeze().tolist()
                    )
                    best_score = confidence
                    best_mask_coords = contour_coords
                    best_mask = mask

    if best_mask_coords is not None:
        contour_np = np.array(best_mask_coords, dtype=np.int32).reshape(-1, 1, 2)

        # Draw green contour
        cv2.drawContours(image, [contour_np], -1, (0, 255, 0), 2)

        # Draw semi-transparent red mask
        red_mask = np.zeros_like(image, dtype=np.uint8)
        red_mask[:, :, 2] = (best_mask * 255).astype(np.uint8)
        alpha = 0.4
        cv2.addWeighted(red_mask, alpha, image, 1 - alpha, 0, image)

        # Add confidence score as text
        cv2.putText(
            image,
            f"Durian: {best_score:.2f}",
            (contour_np[0][0][0], contour_np[0][0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

        # Save annotated image
        annotated_path = os.path.join(
            annotated_dir, f"{os.path.splitext(img_name)[0]}_annotated.png"
        )
        cv2.imwrite(annotated_path, image)

        # Save contour coordinates and confidence
        coords_path = os.path.join(coords_dir, f"{os.path.splitext(img_name)[0]}.txt")
        with open(coords_path, "w") as f:
            f.write(f"Contour: {best_mask_coords}\n")
            f.write(f"Confidence: {best_score}\n")

        print(
            f"Inference completed for {img_name}.\n- Annotated image: {annotated_path}\n- Coordinates saved: {coords_path}"
        )

    else:
        print(f"No durian mask found for {img_name}")
