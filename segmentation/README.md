# 🧠 Segmentation Module (YOLOv8)

This directory contains all code and tools related to **instance segmentation** using the YOLOv8 framework from Ultralytics.

## 📁 Folder Structure

```
segmentation/
├── predict_folder.py    # Predict on a folder of images using a trained model
├── analyse_model/       # Analyze YOLO model performance (metrics, masks, visual outputs)
├── split_train_val.py   # Split dataset into train/val sets (images + labels)
```

## ⚙️ YOLOv8 Training Command

To launch a YOLOv8 **segmentation** training run:

```bash
yolo task=segment mode=train model=yolov8m-seg.pt data=config.yaml epochs=100 imgsz=640 batch=16 name=yolov8m-seg
```

* `task=segment` indicates instance segmentation task
* `model=yolov8m-seg.pt` is the pretrained checkpoint
* `data=config.yaml` defines class names and dataset paths
* `epochs`, `imgsz`, `batch`, etc. are training hyperparameters

## 🔍 YOLOv8 Prediction Command

To run inference on a single image or folder:

```bash
yolo task=segment mode=predict model=runs/segment/yolov8m-seg/weights/best.pt source=path/to/images/ conf=0.5 save=True
```

* `model` should point to your trained weights
* `source` can be a folder or single image
* `conf` is the confidence threshold
* `save=True` saves output images in `runs/segment/predict/`

## ✅ Status

* [x] Image-folder prediction script ready
* [x] Dataset train/val split script operational
* [x] Model evaluation & mask analysis script


