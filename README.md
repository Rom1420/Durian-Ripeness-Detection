# Durian Ripeness Detection 🍈

> **Deep Learning Pipeline for Automated Durian Maturity Detection**

This repository provides a complete end-to-end solution for **detecting** and **classifying** the ripeness of durian fruits. It combines **state-of-the-art object detection** (YOLOv8 & Detectron2) and **custom CNN classification** to assist producers in making better harvesting and export decisions.

---

## 📂 Project Structure

```
Durian-Ripeness-Detection/
├── webapp/          # React + FastAPI frontend/backend (Dockerized) to upload images and view results
├── segmentation/    # YOLOv8 & Detectron2 scripts/configs for fruit segmentation
├── classification/  # CNN training & evaluation for maturity classification
└── README.md
```

* **`webapp/`** – User-friendly interface built with React and FastAPI, packaged with Docker.
* **`segmentation/`** – Detection and cropping of durians using YOLOv8 and Detectron2.
* **`classification/`** – Models to classify maturity stages from cropped/original images.

---

## 🚀 Features

* **Durian Detection** – Accurate bounding box detection using YOLOv8 and Detectron2.
* **Maturity Classification** – CNN trained on multiple datasets (cropped/original/mixed).
* **Web Interface** – Simple Dockerized React + FastAPI app with real-time results.*

---

## 📊 Dataset

* Original, cropped and mixed images of durians.
* Four maturity classes:

  1. **Ripe 1** – No ripe
  2. **Ripe 2** – Mature
  3. **Ripe 3** – Ripe
  4. **Ripe 4** – Overripe

---

## 🛠️ Tech Stack

* **Frontend:** React
* **Backend:** FastAPI
* **Segmentation:** YOLOv8 (Ultralytics), Detectron2
* **Classification:** Custom CNN in PyTorch
* **Other:** Python, OpenCV, Matplotlib, Docker

---

## 📌 Results Overview

* **Best CNN Accuracy:** 93% on cropped images dataset
* **Use Case:** Supports producers in deciding harvest time and export readiness

---

## 👤 Authors

**Romain Abbonato** – *Durian Ripeness Detection Project*
**Luca Del Rosso** – *Durian Ripeness Detection Project*

📧 Contact: [romain.abbonato@etu.unice.fr](romain.abbonato@etu.unice.fr)
