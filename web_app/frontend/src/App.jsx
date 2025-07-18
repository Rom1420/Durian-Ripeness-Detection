import { useState } from "react";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);

    if (selectedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8080/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Erreur pendant la prédiction :", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="title">Durians Ripeness Detection</h1>

      <div className={`upload-box ${isLoading ? "loading" : ""}`}>
        <label
          htmlFor="file-upload"
          className={`btn-grad ${file ? "has-file" : ""}`}
        >
          {file ? file.name : "Choose a file"}
        </label>

        <input
          id="file-upload"
          type="file"
          onChange={handleFileChange}
          className="file-input"
          disabled={isLoading}
        />

        <button
          onClick={handleSubmit}
          disabled={!file || isLoading}
          className="submit-button"
        >
          {isLoading ? "Processing..." : "Send to model"}
        </button>

        {isLoading && <div className="spinner"></div>}
      </div>

      {result?.cnn_prediction_on_original && (
        <div className="prediction-highlight">
          <p>Predicted maturity:</p>
          <h2>
            <strong>{result.cnn_prediction_on_original.class}</strong>{" "}
            <em>({result.cnn_prediction_on_original.confidence})</em>
          </h2>
        </div>
      )}

      {result?.yolo_segmented_image && (
        <div className="image-container">
          <img src={preview} alt="Preview" className="result-image" />
        </div>
      )}
    </div>
  );
}
