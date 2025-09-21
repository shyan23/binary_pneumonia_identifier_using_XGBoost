

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import cv2 as cv
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import shap
import time
import base64

# ============================================================================
# 1. DEFINE APP AND RESPONSE MODELS
# ============================================================================
app = FastAPI(
    title="X-Ray Vision API with Visual Explainability",
    description="An API to predict pneumonia and provide both numeric (SHAP) and visual (highlighted regions) explanations.",
    version="3.0.0"
)

# Pydantic models for structured, predictable API responses
class ExplanationDetail(BaseModel):
    feature: str
    value: float
    contribution: float

class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    label: int
    processing_time_seconds: float
    explanation_image_base64: str  # The new field for our visual explanation
    explanation_numeric: list[ExplanationDetail]

# ============================================================================
# 2. LOAD MODELS AND ASSETS (Done once at startup)
# ============================================================================
model_assets = {}

@app.on_event("startup")
def load_model_assets():
    """Load the model, scaler, and SHAP explainer when the app starts."""
    print("   -> Loading model and scaler...")
    try:
        model = joblib.load("cpu_optimized_model_v4.pkl")
        scaler = joblib.load("cpu_feature_scaler_v4.pkl")
    except FileNotFoundError:
        print("❌ CRITICAL ERROR: Model or scaler files not found.")
        exit()

    model_assets['model'] = model
    model_assets['scaler'] = scaler
    
    model_assets['feature_names'] = [
        'lung_mean_intensity', 'lung_std_intensity', 'lung_entropy',
        'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
        'num_suspicious_regions', 'total_suspicious_area_ratio', 'avg_suspicious_area',
        'avg_region_circularity', 'avg_region_solidity', 'avg_region_intensity',
        'std_region_intensity', 'intensity_contrast'
    ]

    print("   -> Creating SHAP explainer...")
    model_assets['explainer'] = shap.TreeExplainer(model_assets['model'])
    
    print("✅ Model assets loaded successfully!")

# ============================================================================
# 3. REPLICATE THE FEATURE EXTRACTION PIPELINE (Unchanged)
# ============================================================================
def preprocess_image_advanced(image_bytes: bytes):
    try:
        image = cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_GRAYSCALE)
        if image is None: return None
        resized = cv.resize(image, (512, 512))
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        filtered = cv.bilateralFilter(enhanced, 9, 75, 75)
        return filtered
    except Exception:
        return None

def segment_lungs_advanced(image):
    try:
        _, binary_mask = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        closed = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel_close, iterations=2)
        contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours: return image
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        final_mask = np.zeros_like(image)
        for contour in contours[:2]:
            if cv.contourArea(contour) > 10000:
                cv.fillPoly(final_mask, [contour], 255)
        return cv.bitwise_and(image, final_mask)
    except Exception:
        return image

def find_suspicious_regions_advanced(segmented_lung_image):
    if segmented_lung_image is None or not np.any(segmented_lung_image): return []
    try:
        adaptive_thresh = cv.adaptiveThreshold(segmented_lung_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 5)
        contours, _ = cv.findContours(adaptive_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if 100 < cv.contourArea(c) < (segmented_lung_image.shape[0] * segmented_lung_image.shape[1] * 0.1)]
        return filtered_contours
    except Exception:
        return []

def extract_comprehensive_features(image, contours):
    features = {name: 0 for name in model_assets['feature_names']}
    lung_area = np.count_nonzero(image)
    if lung_area == 0: return features
    lung_pixels = image[image > 0]
    lung_mean_intensity = np.mean(lung_pixels)
    features['lung_mean_intensity'] = lung_mean_intensity
    features['lung_std_intensity'] = np.std(lung_pixels)
    hist, _ = np.histogram(lung_pixels, bins=64, density=True)
    features['lung_entropy'] = -np.sum(hist * np.log2(hist + 1e-9))
    try:
        image_8bit = (image / 16).astype(np.uint8)
        glcm = graycomatrix(image_8bit, distances=[1, 3], angles=[0, np.pi/2], levels=16, symmetric=True, normed=True)
        glcm = glcm[1:, 1:, :, :]
        features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
        features['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
        features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
        features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
        features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
    except Exception: pass
    if contours:
        features['num_suspicious_regions'] = len(contours)
        suspicious_areas = [cv.contourArea(c) for c in contours]
        features['total_suspicious_area_ratio'] = sum(suspicious_areas) / lung_area
        features['avg_suspicious_area'] = np.mean(suspicious_areas)
        region_circularity, region_solidity, region_intensities = [], [], []
        for c in contours:
            area, perimeter = cv.contourArea(c), cv.arcLength(c, True)
            if perimeter > 0: region_circularity.append((4 * np.pi * area) / (perimeter ** 2))
            hull_area = cv.contourArea(cv.convexHull(c))
            if hull_area > 0: region_solidity.append(area / hull_area)
            mask = np.zeros(image.shape, dtype=np.uint8); cv.fillPoly(mask, [c], 255)
            region_pixels = image[mask > 0]
            if len(region_pixels) > 0: region_intensities.append(np.mean(region_pixels))
        if region_circularity: features['avg_region_circularity'] = np.mean(region_circularity)
        if region_solidity: features['avg_region_solidity'] = np.mean(region_solidity)
        if region_intensities:
            avg_intensity = np.mean(region_intensities)
            features['avg_region_intensity'] = avg_intensity
            features['std_region_intensity'] = np.std(region_intensities)
            features['intensity_contrast'] = avg_intensity - lung_mean_intensity
    return features

# ============================================================================
# 4. DEFINE THE UNIFIED API ENDPOINT
# ============================================================================

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"message": "Welcome to the X-Ray Vision API! Go to /docs for usage."}


@app.post("/predict/", response_model=PredictionResponse)
async def predict_and_explain(file: UploadFile = File(...)):
    """
    Accepts an X-ray image, runs the full analysis pipeline, and returns
    the diagnosis along with both numeric and VISUAL XAI explanations.
    """
    start_time = time.time()
    
    # === 1. Feature Extraction Pipeline ===
    image_bytes = await file.read()
    preprocessed_image = preprocess_image_advanced(image_bytes)
    if preprocessed_image is None:
        raise HTTPException(status_code=400, detail="Could not read or process the image file.")
    
    segmented_image = segment_lungs_advanced(preprocessed_image)
    regions = find_suspicious_regions_advanced(segmented_image)
    features = extract_comprehensive_features(segmented_image, regions)
    
    features_df = pd.DataFrame([features], columns=model_assets['feature_names'])
    scaled_features = model_assets['scaler'].transform(features_df)
    
    # === 2. Prediction ===
    model = model_assets['model']
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    label = int(prediction)
    confidence = float(probabilities[label])
    diagnosis = "Pneumonia" if label == 1 else "Normal"
    
    # === 3. Visual Explanation (The New Feature!) ===
    # Convert the grayscale segmented image to a color image to draw on
    explanation_image = cv.cvtColor(segmented_image, cv.COLOR_GRAY2BGR)
    
    # Draw the detected suspicious regions in a highly visible color (e.g., green)
    cv.drawContours(explanation_image, regions, -1, (0, 255, 0), 2)
    
    # Encode the image to a JPEG in memory
    _, buffer = cv.imencode('.jpg', explanation_image)
    
    # Convert the buffer to a Base64 string to send in the JSON response
    explanation_image_base64 = base64.b64encode(buffer).decode('utf-8')

    # === 4. Numeric (SHAP) Explanation ===
    explainer = model_assets['explainer']
    shap_values = explainer.shap_values(scaled_features)
    
    explanation_details = []
    for i, feature_name in enumerate(model_assets['feature_names']):
        explanation_details.append({
            "feature": feature_name,
            "value": float(features_df.iloc[0][feature_name]),
            "contribution": float(shap_values[0][i])
        })
    explanation_details.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    end_time = time.time()
    
    # === 5. Assemble and Return the Complete Response ===
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "label": label,
        "processing_time_seconds": round(end_time - start_time, 4),
        "explanation_image_base64": explanation_image_base64,
        "explanation_numeric": explanation_details
    }

# ============================================================================
# 6. ALLOW RUNNING THE SCRIPT DIRECTLY
# ============================================================================
if __name__ == "__main__":
    print("   -> To run the API, use the command:")
    print("   uvicorn main:app --reload")
    uvicorn.run("start:app", host="0.0.0.0", port=8090, reload=True)