# 🧠 Segmentation Module

This module provides two segmentation pipelines:

- **YOLOv8** (bounding box & instance segmentation)
- **Detectron2** (Mask R-CNN-based instance segmentation)

Each subdirectory includes code for training, predicting, and analyzing durian segmentation models.

---

## 📁 Folder Structure

segmentation/
├── yolo/
│ ├── predict_folder.py # Predict on a folder of images using YOLOv8
│ ├── analyse_model/ # Analyze YOLO model (metrics, masks, visuals)
│ ├── split_train_val.py # Split dataset into train/val sets (images + labels)
│
├── detectron/
│ ├── train.py # Train a Detectron2 Mask R-CNN model on COCO-style dataset
│ ├── detect.py # Predict masks and contours using trained model

---

## 🚀 YOLOv8 Workflow

### 🔧 Training

To launch a YOLOv8 segmentation training run:

```bash
yolo task=segment mode=train model=yolov8m-seg.pt data=config.yaml epochs=100 imgsz=640 batch=16 name=yolov8m-seg
```

- `task=segment` : Specifies that the task is segmentation (as opposed to detection or classification).
- `mode=train` : Indicates that this is a training run.
- `model=yolov8m-seg.pt` : Path to the pretrained YOLOv8 segmentation model to fine-tune.
- `data=config.yaml` : YAML file containing `train`, `val` dataset paths and `names` for each class.
- `epochs=100` : Number of training epochs.
- `imgsz=640` : Size of the input images (they are resized to this during training).
- `batch=16` : Batch size during training.
- `name=yolov8m-seg` : Name of the training run (will appear in the `runs/segment/` folder).


Use the `yolo` CLI tool with task set to `segment`, mode as `train`, and specify your pretrained model checkpoint (e.g. yolov8m-seg.pt). Make sure your data YAML file includes `train`, `val` paths and `names` for each class. You can also customize parameters such as `epochs`, `imgsz`, `batch`, and set a name for the run.

### 🖼️ Prediction

To run inference on a single image or a folder of images:

```bash
yolo task=segment mode=predict model=path/to/model.pt source=path/to/images/ conf=0.5 save=True
```

- `task=segment` : Segmentation mode (mandatory).
- `mode=predict` : Indicates you want to run inference (prediction).
- `model=path/to/model.pt` : Path to the trained model weights.
- `source=path/to/images/` : Path to the image file or folder to predict.
- `conf=0.5` : Confidence threshold for mask predictions.
- `save=True` : Save the output images and masks in a subfolder under `runs/segment/predict`.


Set mode to `predict`, provide the path to your trained weights (e.g. runs/segment/.../best.pt), and the source path to your image(s). You can also adjust the confidence threshold (`conf`) and enable saving with `save=True` to output results to `runs/segment/predict`.

---

## 🧪 Detectron2 Workflow

### 🔧 Training

The `train.py` script is used to train a Detectron2 model on a COCO-style dataset.

First, register your dataset using Detectron2’s `register_coco_instances` function. Ensure your annotations are in COCO JSON format and images are correctly structured in `train` and `val` directories.

Configure the training hyperparameters such as learning rate, batch size, max iterations, and output directory. The model will automatically save checkpoints and metrics.

### 🖼️ Inference

Use the `detect.py` script to run predictions on a folder of images.

The script loads the trained Detectron2 model and performs mask prediction and contour extraction for each image. It outputs:
- Annotated images with masks and contours
- .txt files containing polygon coordinates and confidence scores

The results are saved in a structured output folder named according to the checkpoint and timestamp.

---
