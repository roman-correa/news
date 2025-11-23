# Real Estate Price Predictor — Medellín

Do you want to know what price you should ask when buying or selling an apartment or house in large Colombian cities? Welcome aboard!

This project uses **XGBoost**, a powerful machine learning algorithm, to estimate the price per square meter for real estate in **Medellín**. The model can help you:

- Predict sale prices (`venta`)  
- Predict rental prices (`arriendo`)  
- Understand relevant features that influence pricing  

---

## 🧪 Key Features

- **Data-driven**: Trained on real estate data (`.csv` files for rent and sale).  
- **Machine Learning Models**: Uses XGBoost and Random Forest.  
- **Model Evaluation**: Performance metrics and visualizations (e.g. `model_performance.png`) are available.  
- **Geospatial Analysis**: Uses GeoJSON data for Medellín neighborhoods (`medellin.geojson`).  

---

## 📂 Repository Contents

Here’s a summary of the important files/folders:

| File | Description |
|---|---|
| `app.py` | Main application code (probably for running predictions) |
| `arr_mede_final.csv` / `venta_medellin.csv` | Datasets for rental and sale data |
| `preprocessor.pkl` | Preprocessing pipeline for feature engineering |
| `xgb_model_arr_med.pkl` / `xgb_model_ven_med.pkl` | Trained XGBoost models for rent and sale |
| `rf_model_arr_med.pkl` | Random Forest model for rent |
| `best_features_arr.pkl` / `best_features_ven.pkl` | Important features for the models |
| `medellin.geojson` | Geographic data for Medellín neighborhoods |
| `model_performance.png`, `model_performance_arriendo.png`, `model_performance_venta.png` | Model evaluation plots |
| `requirements.txt` | Python dependencies |
| `secrets.toml` | Secret config (e.g. API keys) |

---

## 🚀 How to Use It

1. **Clone the repo**  
   ```bash
   git clone https://github.com/roman-correa/Compra_Venta_Medellin.git
   cd Compra_Venta_Medellin
