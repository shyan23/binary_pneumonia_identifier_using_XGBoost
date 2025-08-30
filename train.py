# ============================================================================
# X-RAY VISION (CPU v4 - FINAL TRAINING SCRIPT)
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# NOTE: This script skips the time-consuming Optuna optimization and uses the
#       pre-discovered best hyperparameters to train the final model directly.
# ============================================================================

print("🩻 X-RAY VISION (CPU v4): Final Model Training")
print("=" * 70)

# (All imports and helper functions from the previous script remain the same)
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import warnings
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import shap
from skimage.feature import graycomatrix, graycoprops

warnings.filterwarnings('ignore')
print("✅ SETUP COMPLETE - All libraries loaded!")

# --- [Paste all your helper functions here] ---
# preprocess_image_advanced, segment_lungs_advanced, find_suspicious_regions_advanced,
# extract_comprehensive_features, process_single_image, load_and_process_data_parallel,
# comprehensive_model_evaluation, interpret_model_with_shap
# ... (for brevity, I am omitting the function code which you already have)

def preprocess_image_advanced(image_path):
    try:
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
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
        if not contours: return image, np.zeros_like(image)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        final_mask = np.zeros_like(image)
        for contour in contours[:2]:
            if cv.contourArea(contour) > 10000:
                cv.fillPoly(final_mask, [contour], 255)
        return cv.bitwise_and(image, final_mask), final_mask
    except Exception:
        return image, np.zeros_like(image)

def find_suspicious_regions_advanced(segmented_lung_image):
    if segmented_lung_image is None or not np.any(segmented_lung_image): return []
    try:
        adaptive_thresh = cv.adaptiveThreshold(segmented_lung_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 5)
        contours, _ = cv.findContours(adaptive_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        image_area = segmented_lung_image.shape[0] * segmented_lung_image.shape[1]
        for contour in contours:
            area = cv.contourArea(contour)
            if 100 < area < image_area * 0.1:
                filtered_contours.append(contour)
        return filtered_contours
    except Exception:
        return []

print("✅ ADVANCED IMAGE PROCESSING FUNCTIONS LOADED!")

def extract_comprehensive_features(image, contours):
    features = {
        'lung_mean_intensity': 0, 'lung_std_intensity': 0, 'lung_entropy': 0,
        'glcm_contrast': 0, 'glcm_dissimilarity': 0, 'glcm_homogeneity': 0,
        'glcm_energy': 0, 'glcm_correlation': 0,
        'num_suspicious_regions': 0, 'total_suspicious_area_ratio': 0, 'avg_suspicious_area': 0,
        'avg_region_circularity': 0, 'avg_region_solidity': 0,
        'avg_region_intensity': 0, 'std_region_intensity': 0,
        'intensity_contrast': 0
    }
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
    except Exception:
        pass
    if contours:
        features['num_suspicious_regions'] = len(contours)
        suspicious_areas = [cv.contourArea(c) for c in contours]
        features['total_suspicious_area_ratio'] = sum(suspicious_areas) / lung_area
        features['avg_suspicious_area'] = np.mean(suspicious_areas)
        region_circularity, region_solidity, region_intensities = [], [], []
        for c in contours:
            area = cv.contourArea(c)
            perimeter = cv.arcLength(c, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                region_circularity.append(circularity)
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                region_solidity.append(solidity)
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv.fillPoly(mask, [c], 255)
            region_pixels = image[mask > 0]
            if len(region_pixels) > 0:
                region_intensities.append(np.mean(region_pixels))
        if region_circularity: features['avg_region_circularity'] = np.mean(region_circularity)
        if region_solidity: features['avg_region_solidity'] = np.mean(region_solidity)
        if region_intensities:
            avg_intensity = np.mean(region_intensities)
            features['avg_region_intensity'] = avg_intensity
            features['std_region_intensity'] = np.std(region_intensities)
            features['intensity_contrast'] = avg_intensity - lung_mean_intensity
    return features
print("✅ 'Neurosurgeon' FEATURE EXTRACTOR LOADED!")

def process_single_image(img_path):
    try:
        preprocessed = preprocess_image_advanced(img_path)
        if preprocessed is None: return None
        segmented, _ = segment_lungs_advanced(preprocessed)
        suspicious_regions = find_suspicious_regions_advanced(segmented)
        features = extract_comprehensive_features(segmented, suspicious_regions)
        return features
    except Exception:
        return None

def load_and_process_data_parallel(data_path, dataset_types=["train"], max_samples_per_class=None):
    print(f"\n📥 LOADING AND PROCESSING DATA FROM: {', '.join(dataset_types)}")
    image_paths, labels = [], []
    for dataset_type in dataset_types:
        base_path = os.path.join(data_path, dataset_type)
        print(f"   -> Reading from folder: {base_path}")
        for category, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            category_path = os.path.join(base_path, category)
            if os.path.exists(category_path):
                files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if max_samples_per_class:
                    files = files[:max_samples_per_class]
                for file in files:
                    image_paths.append(os.path.join(category_path, file))
                    labels.append(label)
                print(f"   ✅ Found {len(files)} {category} images in {dataset_type} set.")
    
    if not image_paths: return pd.DataFrame()
    
    print(f"⚡ EXTRACTING FEATURES IN PARALLEL (using all CPU cores)...")
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_single_image)(path) for path in tqdm(image_paths)
    )
    
    all_features = []
    final_labels = []
    for i, res in enumerate(results):
        if res is not None:
            all_features.append(res)
            final_labels.append(labels[i])
            
    print(f"✅ Feature extraction complete! Successfully processed {len(all_features)} out of {len(image_paths)} images.")
    features_df = pd.DataFrame(all_features)
    features_df['label'] = final_labels
    return features_df

def train_final_model_and_save(best_params, features_df, output_dir='artifacts_v4'):
    print("\n🚂 TRAINING FINAL MODEL on all data with best parameters...")
    os.makedirs(output_dir, exist_ok=True)
    
    final_params = best_params.copy()
    class_counts = features_df['label'].value_counts()
    final_params['scale_pos_weight'] = class_counts[0] / class_counts[1]
    
    X = features_df.drop('label', axis=1).fillna(features_df.median())
    y = features_df['label'].values
    
    scaler = RobustScaler()
    full_X_scaled = scaler.fit_transform(X)
    
    final_model = xgb.XGBClassifier(**final_params, tree_method='hist', objective='binary:logistic', use_label_encoder=False, random_state=42, n_jobs=-1)
    final_model.fit(full_X_scaled, y)
    
    joblib.dump(final_model, os.path.join(output_dir, 'cpu_optimized_model_v4.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'cpu_feature_scaler_v4.pkl'))
    with open(os.path.join(output_dir, 'best_params_v4.json'), 'w') as f:
        json.dump(final_params, f, indent=4)

    print(f"✅ Final model and artifacts saved to '{output_dir}' directory.")
    
    return {'model': final_model, 'scaler': scaler, 'feature_names': X.columns.tolist()}

def comprehensive_model_evaluation(model, scaler, feature_names, test_df):
    import seaborn as sns
    print("\n📊 COMPREHENSIVE MODEL EVALUATION ON TEST SET...")
    if test_df.empty: return
    X_test = test_df.drop('label', axis=1).reindex(columns=feature_names, fill_value=0)
    y_test = test_df['label'].values
    X_test_scaled = scaler.transform(X_test.fillna(X_test.median()))
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 FINAL MODEL PERFORMANCE:\n   Accuracy: {accuracy:.1%}\n   Precision: {precision_score(y_test, y_pred):.1%}\n   Recall: {recall_score(y_test, y_pred):.1%}\n   F1-Score: {f1_score(y_test, y_pred):.1%}")
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia']))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def interpret_model_with_shap(model, scaler, features_df):
    print("\n🧠 INTERPRETING MODEL PREDICTIONS WITH SHAP...")
    X = features_df.drop('label', axis=1).fillna(features_df.median())
    X_sample = X.sample(n=min(500, len(X)), random_state=42)
    X_sample_scaled = scaler.transform(X_sample)
    print("   Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_scaled)
    X_sample_scaled_df = pd.DataFrame(X_sample_scaled, columns=X.columns)
    print("   Displaying SHAP summary plot...")
    shap.summary_plot(shap_values, X_sample_scaled_df, plot_type="dot", show=True)


# ============================================================================
# MAIN EXECUTION BLOCK (STREAMLINED)
# ============================================================================
def main():
    # --- Configuration ---
    DATA_PATH = "chest_xray"
    
    print(f"\n🚀 RUNNING IN FAST TRAINING MODE.")

    # --- Step 1: Define the Proven Best Hyperparameters ---
    proven_best_params = {
        'n_estimators': 1905,
        'learning_rate': 0.04032005131705574,
        'max_depth': 5,
        'subsample': 0.7945065050686154,
        'colsample_bytree': 0.8848983488168604,
        'gamma': 1.0268662374485689,
        'lambda': 0.9328797851286241,
        'alpha': 0.1631871462336305,
        'min_child_weight': 3
    }
    print("✅ Using pre-discovered optimal hyperparameters from Trial 94.")
    
    # --- Step 2: Load and Process Training Data ---
    full_train_df = load_and_process_data_parallel(DATA_PATH, dataset_types=["train", "val"])
    
    if not full_train_df.empty:
        # --- Step 3: Train Final Model and Save Artifacts ---
        final_artifacts = train_final_model_and_save(proven_best_params, full_train_df)

        # --- Step 4: Load Test Data and Evaluate ---
        test_df = load_and_process_data_parallel(DATA_PATH, dataset_types=["test"])
        comprehensive_model_evaluation(**final_artifacts, test_df=test_df)
        
        # --- Step 5: Interpret Model ---
        interpret_model_with_shap(model=final_artifacts['model'], scaler=final_artifacts['scaler'], features_df=full_train_df)
        
        print("\n🎉 SCRIPT COMPLETE! 🎉")

if __name__ == "__main__":
    main()