# YOLOv8 Backend API (FastAPI)

Lightweight backend for object detection using a YOLOv8 model, exposed via a FastAPI interface.

---

## 🚀 Launch the Backend

### 1. Install Dependencies

```bash
pip install fastapi uvicorn opencv-python python-multipart ultralytics
```

### 2. Start the Server

```bash
uvicorn main:app --reload --port 8080
```

The server will be accessible at: `📍 http://localhost:8080`

## 📦 Available Endpoints

### `POST /predict`

* **Description**: Analyzes an image using the YOLOv8 model.

* **Method**: `POST`

* **Content-Type**: `multipart/form-data`

* **Parameter**:

  * `file`: Image file (`.jpg`, `.png`, etc.)

* **Response (JSON)**:

  ```json
  {
    "image_url": "static/results/abc123.jpg"
  }
  ```
