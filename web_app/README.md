# Durian Ripeness Detection 🍈

This project is a full-stack machine learning web app that detects the **ripeness level of a durian** fruit from an uploaded image. It uses **YOLOv8** for fruit detection and a **CNN classifier** for predicting the maturity stage (e.g. No Ripe, Mature, Ripe, Overripe).

---

## 🧠 Technologies Used

### Backend (FastAPI)

* YOLOv8 (Ultralytics) for fruit detection and cropping
* Custom CNN for ripeness classification
* Torch / torchvision
* FastAPI for serving predictions

### Frontend (React)

* React + Vite
* Styled with CSS
* Allows image upload and displays predictions + detected images

### Containerization

* Docker & Docker Compose to run the backend and frontend together

---

## 🚀 How to Run with Docker Compose

### ✅ Prerequisites

* Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* Make sure Docker is running (`docker version` should work)

### 📁 Folder Structure (Example)

```
Durian_Ripeness_Detection/
├── backend/
│   ├── main.py
│   ├── train_cnn.py
│   ├── model/
│       └── cnn_original.pth
│       └── yolov8m.pt
├── frontend/
│   ├── src/
│   ├── index.html
│   └── ...
├── docker-compose.yml
└── README.md
```

### ⚙️ Step-by-Step

1. **Clone the repository** (or copy your project locally)

2. **Navigate to the root folder:**

```bash
cd Durian_Ripeness_Detection
```

3. **Run Docker Compose**

```bash
docker-compose up --build
```

This will:

* Start the **FastAPI backend** on `http://localhost:8080`
* Start the **React frontend** on `http://localhost:5173`

4. **Open your browser and test:**

👉 Visit: [http://localhost:5173](http://localhost:5173)

Upload an image and get prediction results!

---

## 🧠 Future Improvements

* Add multi-language support (e.g., Vietnamese/English)
* Model switching maybe 
* Create the Mobile App
* Deploy on the web 

---