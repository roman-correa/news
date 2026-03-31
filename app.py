"""
Compra/Venta Medellín — Real Estate Price Predictor
Author: Roman Alejandro Correa
"""
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
from pathlib import Path
import datetime
import requests as _requests

# Artifacts can live in ./artifacts/ (after train.py) or ./ (legacy flat layout)


def _artifact(name: str) -> str:
    p = Path("artifacts") / name
    return str(p) if p.exists() else name


# ─────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Propiedades Medellín",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Language support
# ─────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "es"

TRANSLATIONS = {
    "es": {
        # Navigation
        "title": "🏙️ Medellín RE",
        "nav": "Navegación",
        "nav_predictor": "🔮 Predictor",
        "nav_explore": "📊 Explorar datos",
        "nav_map": "🗺️ Mapa de oportunidades",
        "filters": "Filtros globales",
        "barrio": "Barrio",
        "min_price": "Precio mínimo (COP)",
        "max_price": "Precio máximo (COP)",
        "all": "Todos",
        "model_status": "Estado del modelo",
        "data_updated": "Datos actualizados",
        "by": "Por Roman Alejandro Correa",

        # Predictor page
        "predictor_title": "Predictor de precios",
        "predictor_desc": "Estima el precio de arriendo o venta de una propiedad en Medellín.",
        "transaction_type": "Tipo de transacción",
        "rental": "Arriendo",
        "sale": "Venta",
        "model_accuracy": "Precisión del modelo",
        "excellent": "Excelente",
        "good": "Bueno",
        "moderate": "Moderado",
        "evaluated_test": "· evaluado en conjunto de prueba (log-space)",

        "property_data": "Datos de la propiedad",
        "area": "Área (m²)",
        "bedrooms": "Habitaciones",
        "bathrooms": "Baños",
        "parking": "Parqueaderos",
        "strata": "Estrato",
        "admin_fee": "Admon. mensual (COP)",
        "metro_distance": "Dist. metro aprox. (km)",
        "predict_btn": "Predecir precio →",

        "result": "Resultado",
        "calculating": "Calculando...",
        "estimated_price": "Precio estimado",
        "probable_range": "Rango probable",
        "price_m2": "Precio / m²",
        "your_property": "Tu propiedad",
        "median_barrio_m2": "Mediana barrio / m²",
        "key_factors": "Factores clave del precio",
        "similar_props": "Propiedades similares en el barrio",
        "no_comparables": "No hay comparables con filtros exactos en este barrio.",
        "fill_form": "Completa el formulario y presiona",
        "barrio_context": "Contexto del barrio seleccionado",
        "median_rental": "Arriendo — mediana",
        "median_sale": "Venta — mediana",
        "listings": "listings",

        # Disclaimer
        "disclaimer_title": "⚠️ Sobre las discrepancias de barrio",
        "disclaimer_important": "Importante:",
        "disclaimer_p1": "Este modelo utiliza coordenadas GPS exactas de las propiedades para hacer predicciones. Sin embargo, el barrio mostrado en la descripción de metrocuadrado a veces no coincide con la ubicación real de la propiedad.",
        "disclaimer_p2": "Esto ocurre porque:",
        "disclaimer_reason1": "Errores de descripción: Algunos vendedores ingresan un barrio incorrecto en el anuncio.",
        "disclaimer_reason2": "Límites de barrios difusos: Las propiedades en los límites pueden estar registradas en un barrio pero ubicarse físicamente en otro.",
        "disclaimer_reason3": "Nombres alternativos: Barrios con múltiples nombres (ej: \"La Candelaria\" vs \"Las Palmas\").",
        "disclaimer_p3": "¿Cómo usamos esto?",
        "disclaimer_p3_desc": "El modelo se basa en las coordenadas reales (GPS), no en la descripción del vendedor. Por lo tanto, si ves que el barrio predicho es diferente, confía en la ubicación GPS — es más preciso.",

        # Explorer page
        "explorer_title": "Explorador de datos",
        "dataset": "Dataset",
        "total_listings": "Total listings",
        "filtered": "filtrados",
        "price_median": "Precio mediana",
        "area_median": "Área mediana",
        "m2": "m²",
        "opportunities": "Oportunidades",
        "underpriced": ">20% subvaloradas",
        "visualizations": "Visualizaciones",
        "price_distribution": "Distribución de precios",
        "price_vs_area": "Precio vs Área",
        "by_barrio": "Por barrio",
        "top_barrios": "Top 25 barrios por precio mediano",
        "data_table": "Tabla de datos",
        "download_csv": "⬇ Descargar CSV filtrado",
        "type_col": "Tipo",
        "price_col": "Precio",
        "area_col": "Área",
        "rooms_col": "Habitaciones",
        "baths_col": "Baños",
        "parking_col": "Parqueaderos",

        # Map page
        "map_title": "Mapa de oportunidades",
        "map_desc": "Barrios y propiedades donde el precio de mercado está por debajo del valor predicho por el modelo.",
        "view": "Ver",
        "found": "encontradas",
        "poi_nearby": "Puntos de interés cercanos",
        "show_on_map": "Mostrar en el mapa",
        "categories": "Categorías",
        "searching_poi": "Buscando puntos de interés...",
        "no_coords": "Sin coordenadas disponibles.",
        "select_row": "Selecciona una fila para ver la propiedad y sus POIs.",
        "real_vs_estimated": "Precio real vs. precio estimado por el modelo",
        "fair_price": "Precio justo",
        "download_opportunities": "⬇ Descargar oportunidades",
        "barrio_col": "Barrio",
        "real_price_col": "Precio real",
        "model_price_col": "Precio modelo",
        "discount_col": "Descuento",
        "metro_col": "Dist. metro",
        "link_col": "Link",
        "filter_by_barrio": "Filtrar por barrio",
        "properties_found": "Propiedades oportunidad",
        "open_link": "Ver →",

        # POI categories
        "restaurants": "🍽️ Restaurantes",
        "supermarkets": "🛒 Supermercados",
        "mall": "🏬 Centro comercial",
        "health": "🏥 Salud",
        "education": "🏫 Educación",
        "transport": "🚇 Transporte",
        "gym": "💪 Gimnasios",
        "parks": "🌳 Parques",
        "banks": "🏦 Bancos/ATM",
    },
    "en": {
        # Navigation
        "title": "🏙️ Medellín RE",
        "nav": "Navigation",
        "nav_predictor": "🔮 Price Predictor",
        "nav_explore": "📊 Explore Data",
        "nav_map": "🗺️ Opportunity Map",
        "filters": "Global Filters",
        "barrio": "Neighborhood",
        "min_price": "Minimum Price (COP)",
        "max_price": "Maximum Price (COP)",
        "all": "All",
        "model_status": "Model Status",
        "data_updated": "Data Updated",
        "by": "By Roman Alejandro Correa",

        # Predictor page
        "predictor_title": "Price Predictor",
        "predictor_desc": "Estimate the rental or sale price of a property in Medellín.",
        "transaction_type": "Transaction Type",
        "rental": "Rental",
        "sale": "Sale",
        "model_accuracy": "Model Accuracy",
        "excellent": "Excellent",
        "good": "Good",
        "moderate": "Moderate",
        "evaluated_test": "· evaluated on test set (log-space)",

        "property_data": "Property Data",
        "area": "Area (m²)",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "parking": "Parking Spaces",
        "strata": "Strata",
        "admin_fee": "Monthly Admin Fee (COP)",
        "metro_distance": "Approx. Metro Distance (km)",
        "predict_btn": "Predict Price →",

        "result": "Result",
        "calculating": "Calculating...",
        "estimated_price": "Estimated Price",
        "probable_range": "Probable Range",
        "price_m2": "Price / m²",
        "your_property": "Your Property",
        "median_barrio_m2": "Neighborhood Median / m²",
        "key_factors": "Key Price Factors",
        "similar_props": "Similar Properties in the Neighborhood",
        "no_comparables": "No comparables found with exact filters in this neighborhood.",
        "fill_form": "Complete the form and press",
        "barrio_context": "Selected Neighborhood Context",
        "median_rental": "Rental — Median",
        "median_sale": "Sale — Median",
        "listings": "listings",

        # Disclaimer
        "disclaimer_title": "⚠️ About Neighborhood Discrepancies",
        "disclaimer_important": "Important:",
        "disclaimer_p1": "This model uses exact GPS coordinates of properties to make predictions. However, the neighborhood shown in the metrocuadrado listing sometimes does not match the actual property location.",
        "disclaimer_p2": "This happens because:",
        "disclaimer_reason1": "Description Errors: Some sellers enter an incorrect neighborhood in the listing.",
        "disclaimer_reason2": "Fuzzy Boundaries: Properties on neighborhood borders may be registered in one area but physically located in another.",
        "disclaimer_reason3": "Alternative Names: Neighborhoods with multiple names (e.g., \"La Candelaria\" vs \"Las Palmas\").",
        "disclaimer_p3": "How We Handle This:",
        "disclaimer_p3_desc": "The model is based on real GPS coordinates, not the seller's description. Therefore, if you see that the predicted neighborhood differs, trust the GPS location — it's more accurate.",

        # Explorer page
        "explorer_title": "Data Explorer",
        "dataset": "Dataset",
        "total_listings": "Total Listings",
        "filtered": "filtered",
        "price_median": "Median Price",
        "area_median": "Median Area",
        "m2": "m²",
        "opportunities": "Opportunities",
        "underpriced": ">20% underpriced",
        "visualizations": "Visualizations",
        "price_distribution": "Price Distribution",
        "price_vs_area": "Price vs Area",
        "by_barrio": "By Neighborhood",
        "top_barrios": "Top 25 Neighborhoods by Median Price",
        "data_table": "Data Table",
        "download_csv": "⬇ Download Filtered CSV",
        "type_col": "Type",
        "price_col": "Price",
        "area_col": "Area",
        "rooms_col": "Bedrooms",
        "baths_col": "Bathrooms",
        "parking_col": "Parking",

        # Map page
        "map_title": "Opportunity Map",
        "map_desc": "Neighborhoods and properties where the market price is below the value predicted by the model.",
        "view": "View",
        "found": "found",
        "poi_nearby": "Nearby Points of Interest",
        "show_on_map": "Show on Map",
        "categories": "Categories",
        "searching_poi": "Searching for points of interest...",
        "no_coords": "No coordinates available.",
        "select_row": "Select a row to see the property and its POIs.",
        "real_vs_estimated": "Real Price vs Model Estimated Price",
        "fair_price": "Fair Price",
        "download_opportunities": "⬇ Download Opportunities",
        "barrio_col": "Neighborhood",
        "real_price_col": "Real Price",
        "model_price_col": "Model Price",
        "discount_col": "Discount",
        "metro_col": "Metro Distance",
        "link_col": "Link",
        "filter_by_barrio": "Filter by Neighborhood",
        "properties_found": "Opportunity Properties",
        "open_link": "View →",

        # POI categories
        "restaurants": "🍽️ Restaurants",
        "supermarkets": "🛒 Supermarkets",
        "mall": "🏬 Shopping Mall",
        "health": "🏥 Health",
        "education": "🏫 Education",
        "transport": "🚇 Transport",
        "gym": "💪 Gyms",
        "parks": "🌳 Parks",
        "banks": "🏦 Banks/ATM",
    }
}


def t(key: str) -> str:
    """Translate a key using the current language."""
    return TRANSLATIONS[st.session_state.lang].get(key, key)


# ─────────────────────────────────────────────
# Theme / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0c1220;
    color: #dde6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #1e2d42;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #111e30 0%, #0e1828 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 12px;
    font-weight: 500;
    color: #5f8ab0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 28px;
    font-weight: 700;
    color: #e8f2ff;
    font-family: 'DM Mono', monospace;
}
.metric-card .sub {
    font-size: 13px;
    color: #5f8ab0;
    margin-top: 4px;
}

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, #0d2137 0%, #0a1c2e 100%);
    border: 1px solid #1a7a6e;
    border-left: 4px solid #0fd4c0;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
}
.pred-box .pred-label {
    font-size: 12px;
    font-weight: 600;
    color: #0fd4c0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.pred-box .pred-value {
    font-size: 38px;
    font-weight: 700;
    color: #ffffff;
    font-family: 'DM Mono', monospace;
}
.pred-box .pred-range {
    font-size: 13px;
    color: #5f8ab0;
    margin-top: 6px;
}

/* Feature bar */
.feat-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.feat-name {
    width: 160px;
    font-size: 12px;
    color: #7a9ab8;
}
.feat-bar-wrap {
    flex: 1;
    background: #0e1828;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.feat-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #0fd4c0, #0ea5e9);
}
.feat-val {
    width: 60px;
    text-align: right;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    color: #a0c0dc;
}

/* Section headers */
.section-header {
    font-size: 11px;
    font-weight: 600;
    color: #3a6080;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #1a2d42;
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
}

/* Disclaimer box */
.disclaimer-box {
    background: linear-gradient(135deg, #0d2542 0%, #0a1f35 100%);
    border: 1px solid #0fd4c0;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    color: #a0c0dc;
    font-size: 14px;
    line-height: 1.8;
}
.disclaimer-box strong {
    color: #0fd4c0;
}
.disclaimer-box ul {
    margin: 16px 0 16px 24px;
}
.disclaimer-box li {
    margin: 10px 0;
}

/* Opportunity badge */
.badge-opp {
    display: inline-block;
    background: #0a2a1e;
    border: 1px solid #0f7a50;
    color: #2ecc8a;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
}
.badge-norm {
    display: inline-block;
    background: #1a1e30;
    border: 1px solid #2a3a55;
    color: #5f8ab0;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Language toggle in top right
# ─────────────────────────────────────────────
col_spacer, col_lang = st.columns([5, 1])
with col_lang:
    lang_cols = st.columns(2)
    with lang_cols[0]:
        if st.button("🇪🇸 ES", use_container_width=True,
                     type="primary" if st.session_state.lang == "es" else "secondary"):
            st.session_state.lang = "es"
            st.rerun()
    with lang_cols[1]:
        if st.button("🇬🇧 EN", use_container_width=True,
                     type="primary" if st.session_state.lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()


# ─────────────────────────────────────────────
# StackedEnsemble — must be defined here so pickle can deserialise it
# (pickle looks up the class in the module where it's being loaded)
# ─────────────────────────────────────────────


class StackedEnsemble:
    """XGBoost + LightGBM → Ridge meta-learner."""

    def __init__(self, xgb_p=None, lgb_p=None, cat_p=None):
        # cat_p kept for backward compat with old pickles — ignored
        self.xgb_p = xgb_p or {}
        self.lgb_p = lgb_p or {}
        self.base_ = []
        self.meta_ = None

    def predict(self, X) -> np.ndarray:
        X = X.replace([np.inf, -np.inf], np.nan)
        medians = X.median().fillna(0)
        X = X.fillna(medians)
        return self.meta_.predict(
            np.column_stack([m.predict(X) for m in self.base_])
        )


# ─────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────
def load_pickle(path: str):
    """Load a pickle file with a clear error if the file is missing or corrupt."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Run train.py to generate artifacts, then commit the artifacts/ folder."
        )
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_geojson(path: str):
    return gpd.read_file(path)


# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
try:
    arr_cats = load_pickle(_artifact("best_features_arr.pkl"))
    ven_cats = load_pickle(_artifact("best_features_ven.pkl"))
    list_barrios = load_pickle(_artifact("list_barrios.pkl"))
    ppmc_arr = load_pickle(_artifact("price_per_m2_arr.pkl"))
    ppmc_ven = load_pickle(_artifact("price_per_m2_ven.pkl"))
    pppz_arr = load_pickle(_artifact("price_per_space_arr.pkl"))
    pppz_ven = load_pickle(_artifact("price_per_space_ven.pkl"))
    pppp_arr = load_pickle(_artifact("price_per_parking_arr.pkl"))
    pppp_ven = load_pickle(_artifact("price_per_parking_ven.pkl"))
    barrio_te_arr = load_pickle(_artifact("barrio_te_arr.pkl"))
    barrio_te_ven = load_pickle(_artifact("barrio_te_ven.pkl"))

    _pp_arr_path = _artifact("preprocessor_arr.pkl")
    _pp_ven_path = _artifact("preprocessor_ven.pkl")
    _pp_legacy = _artifact("preprocessor.pkl")
    preprocessor_arr = load_pickle(_pp_arr_path if Path(
        _pp_arr_path).exists() else _pp_legacy)
    preprocessor_ven = load_pickle(_pp_ven_path if Path(
        _pp_ven_path).exists() else _pp_legacy)

    # Main model — prefer stacked ensemble, fall back to XGBoost
    _has_stack = Path(_artifact("stack_arr.pkl")).exists()
    if _has_stack:
        model_arr = load_pickle(_artifact("stack_arr.pkl"))
        model_ven = load_pickle(_artifact("stack_ven.pkl"))
    else:
        model_arr = load_pickle(_artifact("xgb_model_arr_med.pkl"))
        model_ven = load_pickle(_artifact("xgb_model_ven_med.pkl"))

    # Quantile models — prefer real, fall back to ±12% heuristic
    _has_quantile = Path(_artifact("q10_arr.pkl")).exists()
    if _has_quantile:
        q10_arr = load_pickle(_artifact("q10_arr.pkl"))
        q90_arr = load_pickle(_artifact("q90_arr.pkl"))
        q10_ven = load_pickle(_artifact("q10_ven.pkl"))
        q90_ven = load_pickle(_artifact("q90_ven.pkl"))
    else:
        q10_arr = q90_arr = q10_ven = q90_ven = None

    # Saved R² from train.py
    _r2_path = _artifact("model_r2.pkl")
    _saved_r2 = load_pickle(_r2_path) if Path(_r2_path).exists() else None

except Exception as _e:
    _saved_r2 = None
    _has_stack = False
    _has_quantile = False
    model_arr = model_ven = None
    q10_arr = q90_arr = q10_ven = q90_ven = None
    arr_cats = ven_cats = []
    preprocessor_arr = preprocessor_ven = None
    import traceback
    st.error(
        f"**Error loading model artifacts.**\n\n"
        f"`{type(_e).__name__}: {_e}`\n\n"
        f"```\n{traceback.format_exc()}\n```"
    )
    st.stop()

loan = load_csv(_artifact("arr_mede_final.csv"))
sales = load_csv(_artifact("ven_mede_final.csv"))
gdf = load_geojson("medellin.geojson")

# ─────────────────────────────────────────────
# Constants (must match train.py exactly)
# ─────────────────────────────────────────────
NUM_ATTRIBS = [
    "area", "habitaciones", "baños", "parqueaderos", "espacios",
    "axe", "axh", "axa", "parq2", "garaje_bin",
    "estrato", "administracion", "dist_metro_km",
    "ppmc", "pppz", "pppp", "new_index",
    "pppp/pppz", "pppp/ppmc", "pppz/ppmc",
    "barrio_count",
    "barrio_te",
]
CAT_ATTRIBS = ["tipo"]

if _saved_r2:
    MODEL_R2 = _saved_r2
else:
    @st.cache_data(ttl=3600)
    def _compute_r2() -> dict[str, float]:
        from sklearn.metrics import r2_score
        results = {}
        for kind, df, pp, m, cats in [
            ("arr", loan,  preprocessor_arr, model_arr, arr_cats),
            ("ven", sales, preprocessor_ven, model_ven, ven_cats),
        ]:
            try:
                if pp is None or m is None:
                    results[kind] = float("nan")
                    continue
                # Only use columns the preprocessor actually knows about
                pp_cols = [c for c in NUM_ATTRIBS +
                           CAT_ATTRIBS if c in df.columns]
                X = pd.DataFrame(pp.transform(df[pp_cols]),
                                 columns=pp.get_feature_names_out())
                avail = [c for c in cats if c in X.columns]
                if not avail:
                    results[kind] = float("nan")
                    continue
                preds = m.predict(X[avail])
                results[kind] = round(
                    float(r2_score(np.log1p(df["precio"]), preds)), 4)
            except Exception:
                results[kind] = float("nan")
        return results
    MODEL_R2 = _compute_r2()


def _data_date() -> str:
    """Return the modification date of the arriendo CSV as a human string."""
    try:
        p = Path(_artifact("arr_mede_final.csv"))
        ts = p.stat().st_mtime
        return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M")
    except Exception:
        return "desconocida" if st.session_state.lang == "es" else "unknown"


DATA_DATE = _data_date()

# barrio_norm = lowercase version of nombre, used only for comparisons
# nombre itself now keeps original shapefile casing (e.g. "El Poblado")
for df in [loan, sales]:
    df["barrio_norm"] = df["nombre"].astype(str).str.strip().str.lower()


# ─────────────────────────────────────────────
# Opportunity labels (computed once at startup)
# ─────────────────────────────────────────────
def _add_derived_features(df: pd.DataFrame, te_map: pd.Series) -> pd.DataFrame:
    """
    Compute features that are not stored in the CSV because they require
    train-only target encoding. Must mirror train.py's prep_dataset exactly.
    """
    df = df.copy()

    # Smoothed target encoding — use the map saved by train.py (train rows only)
    global_mean = float(te_map.mean())
    df["barrio_te"] = df["nombre"].map(te_map).fillna(global_mean)

    # barrio_count is safe (not price-derived) — already in the CSV from engineer_features
    # so no need to recompute it here

    return df


@st.cache_data(ttl=3600)
def add_opportunity_labels(_loan, _sales):
    """Return loan and sales DataFrames with opportunity columns added."""
    loan_c = _loan.copy()
    sales_c = _sales.copy()

    # Guard: if models or preprocessors aren't loaded, skip silently
    if preprocessor_arr is None or preprocessor_ven is None or model_arr is None or model_ven is None:
        for df in [loan_c, sales_c]:
            df["cat_pred"] = np.nan
            df["is_underpriced"] = False
            df["pct_underpriced"] = 0.0
            df["oportunity_houses"] = False
        return loan_c, sales_c

    loan_c = _add_derived_features(loan_c,  barrio_te_arr)
    sales_c = _add_derived_features(sales_c, barrio_te_ven)

    def _opp(df, pp, model, cats):
        # Only pass columns the preprocessor was trained on
        pp_cols = [c for c in NUM_ATTRIBS + CAT_ATTRIBS if c in df.columns]
        std = pd.DataFrame(pp.transform(
            df[pp_cols]), columns=pp.get_feature_names_out())
        avail = [c for c in cats if c in std.columns]
        df["cat_pred"] = np.expm1(model.predict(std[avail]))
        df["is_underpriced"] = df["precio"] < df["cat_pred"]
        df["pct_underpriced"] = (
            df["cat_pred"] - df["precio"]) / df["cat_pred"] * 100
        df["oportunity_houses"] = df["pct_underpriced"] > 20
        return df

    loan_c = _opp(loan_c,  preprocessor_arr, model_arr, arr_cats)
    sales_c = _opp(sales_c, preprocessor_ven, model_ven, ven_cats)
    return loan_c, sales_c


loan, sales = add_opportunity_labels(loan, sales)


# ─────────────────────────────────────────────
# Feature engineering helper
# ─────────────────────────────────────────────
def _build_input_df(area, habitaciones, banos, parqueaderos, barrio, tipo, kind="arr",
                    estrato: float = 3.0, administracion: float = 0.0,
                    dist_metro_km: float = 1.0) -> pd.DataFrame:
    """
    Return a single-row DataFrame with exactly NUM_ATTRIBS + CAT_ATTRIBS columns
    so it can be passed directly to preprocessor.transform().
    """
    ppmc = ppmc_arr if kind == "arr" else ppmc_ven
    pppz = pppz_arr if kind == "arr" else pppz_ven
    pppp = pppp_arr if kind == "arr" else pppp_ven

    espacios = habitaciones + parqueaderos + banos
    axe = area / espacios if espacios else np.nan
    axh = area / habitaciones if habitaciones else np.nan
    axa = area ** 2
    parq2 = parqueaderos ** 2

    barrio_ppmc = ppmc.get(barrio, np.nan)
    barrio_pppz = pppz.get(barrio, np.nan)
    barrio_pppp = pppp.get(barrio, np.nan)

    def safe_ratio(a, b):
        try:
            return float(a / b) if b and not np.isnan(float(b)) and b != 0 else 1.0
        except Exception:
            return 1.0

    ppmc_max = float(ppmc.max()) if hasattr(ppmc, "max") and len(ppmc) else 1.0
    new_index = (barrio_ppmc / ppmc_max * 100) if ppmc_max else 0.0

    # Target encoding — smoothed, falls back to global mean for unseen barrios
    te_map = barrio_te_arr if kind == "arr" else barrio_te_ven
    barrio_te = float(te_map.get(barrio, te_map.mean()))

    # Listing density for the barrio
    ref = loan if kind == "arr" else sales
    barrio_count = int((ref["barrio_norm"] == barrio.strip().lower()).sum())

    # Keys must match NUM_ATTRIBS order exactly
    row = {
        "area":              area,
        "habitaciones":      habitaciones,
        "baños":             banos,
        "parqueaderos":      parqueaderos,
        "espacios":          espacios,
        "axe":               axe,
        "axh":               axh,
        "axa":               axa,
        "parq2":             parq2,
        "garaje_bin":        1 if parqueaderos > 0 else 0,
        "estrato":           estrato,
        "administracion":    administracion,
        "dist_metro_km":     dist_metro_km,
        "ppmc":              barrio_ppmc,
        "pppz":              barrio_pppz,
        "pppp":              barrio_pppp,
        "new_index":         new_index,
        "pppp/pppz":         safe_ratio(barrio_pppp, barrio_pppz),
        "pppp/ppmc":         safe_ratio(barrio_pppp, barrio_ppmc),
        "pppz/ppmc":         safe_ratio(barrio_pppz, barrio_ppmc),
        "barrio_count":      float(barrio_count),
        "barrio_te":         barrio_te,
        "tipo":              tipo,
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────
# Prediction functions
# ─────────────────────────────────────────────
def _validate(barrio, tipo, reference_df) -> str | None:
    barrios_lc = [b.strip().lower() for b in list_barrios]
    if barrio.strip().lower() not in barrios_lc:
        return f'{t("barrio")} "{barrio}" {t("no_in_list") if st.session_state.lang == "es" else "not in list"}.'
    tipos_lc = [t_val.strip().lower()
                for t_val in reference_df["tipo"].unique()]
    if tipo.strip().lower() not in tipos_lc:
        return f'{t("transaction_type")} "{tipo}" {t("no_in_list") if st.session_state.lang == "es" else "not in list"}.'
    return None


def predict(area, habitaciones, banos, parqueaderos, barrio, tipo, kind="arr", **kwargs) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """Return (prediction, price_per_m2, low, high, error_string)."""
    ref_df = loan if kind == "arr" else sales
    err = _validate(barrio, tipo, ref_df)
    if err:
        return None, None, None, None, err

    input_df = _build_input_df(area, habitaciones, banos, parqueaderos, barrio, tipo, kind,
                               estrato=kwargs.get("estrato", 3.0),
                               administracion=kwargs.get(
                                   "administracion", 0.0),
                               dist_metro_km=kwargs.get("dist_metro_km", 1.0))

    pp = preprocessor_arr if kind == "arr" else preprocessor_ven
    cats = arr_cats if kind == "arr" else ven_cats
    model = model_arr if kind == "arr" else model_ven
    q10 = (q10_arr if kind == "arr" else q10_ven) if _has_quantile else None
    q90 = (q90_arr if kind == "arr" else q90_ven) if _has_quantile else None

    X = pp.transform(input_df[NUM_ATTRIBS + CAT_ATTRIBS])
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = pd.DataFrame(X, columns=pp.get_feature_names_out())

    missing = [c for c in cats if c not in X.columns]
    if missing:
        return None, None, None, None, f"Missing columns: {missing}"

    price = float(np.expm1(model.predict(X[cats]))[0])
    ppm2 = price / area if area else None

    if q10 is not None:
        low = float(np.expm1(q10.predict(X[cats]))[0])
        high = float(np.expm1(q90.predict(X[cats]))[0])
    else:
        low, high = price * 0.88, price * 1.12

    return price, ppm2, low, high, None


# ─────────────────────────────────────────────
# Helper: neighbourhood context
# ─────────────────────────────────────────────
def get_barrio_stats(barrio: str) -> dict:
    """Return a dict of median/count stats for a neighbourhood."""
    bn = barrio.strip().lower()
    arr_b = loan[loan["barrio_norm"] == bn]
    ven_b = sales[sales["barrio_norm"] == bn]
    return {
        "arr_count":  len(arr_b),
        "ven_count":  len(ven_b),
        "arr_median": arr_b["precio"].median() if len(arr_b) else None,
        "ven_median": ven_b["precio"].median() if len(ven_b) else None,
        "arr_ppm2":   arr_b["precio"].div(arr_b["area"].replace(0, np.nan)).median() if "area" in arr_b else None,
        "ven_ppm2":   ven_b["precio"].div(ven_b["area"].replace(0, np.nan)).median() if "area" in ven_b else None,
        "arr_opp_pct": arr_b["oportunity_houses"].mean() * 100 if len(arr_b) else 0,
        "ven_opp_pct": ven_b["oportunity_houses"].mean() * 100 if len(ven_b) else 0,
    }


def fmt_price(v: float | None, fallback="—") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    if v >= 1_000_000_000:
        return f"${v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    return f"${v:,.0f}"


# ─────────────────────────────────────────────
# POI fetcher (Overpass / OpenStreetMap) — no API key needed
# ─────────────────────────────────────────────
POI_CATEGORIES = {
    "🍽️ Restaurantes": '[amenity~"restaurant|cafe|fast_food"]',
    "🛒 Supermercados": '[shop~"supermarket|convenience"]',
    "🏬 Centro comercial": '[shop="mall"]',
    "🏥 Salud": '[amenity~"hospital|clinic|pharmacy|dentist"]',
    "🏫 Educación": '[amenity~"school|university|college|library"]',
    "🚇 Transporte": '[amenity~"bus_station|taxi"][public_transport~"station|stop_area"]',
    "💪 Gimnasios": '[leisure~"fitness_centre|sports_centre"]',
    "🌳 Parques": '[leisure="park"]',
    "🏦 Bancos/ATM": '[amenity~"bank|atm"]',
}

POI_COLORS = {
    "🍽️ Restaurantes":    "#f97316",
    "🛒 Supermercados":   "#a78bfa",
    "🏬 Centro comercial": "#fb923c",
    "🏥 Salud":           "#f87171",
    "🏫 Educación":       "#60a5fa",
    "🚇 Transporte":      "#facc15",
    "💪 Gimnasios":       "#4ade80",
    "🌳 Parques":         "#86efac",
    "🏦 Bancos/ATM":      "#94a3b8",
}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_pois(lat: float, lon: float, radius_m: int = 800) -> dict[str, pd.DataFrame]:
    """
    Query Overpass API for POIs around a coordinate.
    Returns dict of category → DataFrame(name, lat, lon).
    Falls back to empty dicts on any network error.
    """
    results = {}
    for cat, tag_filter in POI_CATEGORIES.items():
        query = f"""
        [out:json][timeout:10];
        node{tag_filter}(around:{radius_m},{lat},{lon});
        out body;
        """
        try:
            r = _requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=12,
            )
            elements = r.json().get("elements", [])
            rows = []
            for e in elements:
                if "lat" not in e or "lon" not in e:
                    continue
                tags = e.get("tags", {})
                name = tags.get("name", cat.split()[-1])
                row = {"name": name, "lat": e["lat"], "lon": e["lon"]}
                # OSM rating fields (present on well-mapped venues)
                rating = tags.get("stars") or tags.get(
                    "rating") or tags.get("opening_hours:rating")
                if rating:
                    try:
                        row["rating"] = float(
                            str(rating).replace(",", ".").strip())
                    except ValueError:
                        row["rating"] = None
                else:
                    row["rating"] = None
                # Extra context
                row["cuisine"] = tags.get("cuisine", "")
                row["opening"] = tags.get("opening_hours", "")
                rows.append(row)
            if rows:
                results[cat] = pd.DataFrame(rows)
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {t('title')}")
    st.markdown(f"<div class='section-header'>{t('nav')}</div>",
                unsafe_allow_html=True)

    mode_options = [t('nav_predictor'), t('nav_explore'), t('nav_map')]
    mode = st.radio(
        "Sección",
        mode_options,
        label_visibility="collapsed",
    )

    st.markdown(f"<div class='section-header'>{t('filters')}</div>",
                unsafe_allow_html=True)
    selected_barrio = st.selectbox(
        t('barrio'), [t('all')] + sorted(list_barrios))
    min_price = st.number_input(
        t('min_price'), value=0, step=1_000_000, format="%d")
    max_price = st.number_input(
        t('max_price'), value=2_000_000_000, step=1_000_000, format="%d")

    st.markdown("---")
    st.markdown(f"<div class='section-header'>{t('model_status')}</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"""<div style='font-size:12px;color:#5f8ab0;line-height:2;'>
            📅 <b style='color:#8aabcc'>{t('data_updated')}</b><br>
            <span style='font-family:"DM Mono",monospace;color:#0fd4c0'>{DATA_DATE}</span><br><br>
            📐 <b style='color:#8aabcc'>R² {t('rental')}</b><br>
            <span style='font-family:"DM Mono",monospace;color:#0fd4c0'>{MODEL_R2.get("arr", float("nan")):.4f}</span><br><br>
            📐 <b style='color:#8aabcc'>R² {t('sale')}</b><br>
            <span style='font-family:"DM Mono",monospace;color:#0fd4c0'>{MODEL_R2.get("ven", float("nan")):.4f}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(t('by'))


# ─────────────────────────────────────────────
# PAGE: PREDICTOR
# ─────────────────────────────────────────────
if mode == t('nav_predictor'):
    st.markdown(f"## {t('predictor_title')}")
    st.markdown(t('predictor_desc'))

    # Disclaimer about neighborhood discrepancies
    st.markdown(f"### {t('disclaimer_title')}")

    with st.container(border=True):
        st.markdown(f"**{t('disclaimer_important')}** {t('disclaimer_p1')}")

        st.markdown(f"**{t('disclaimer_p2')}**")

        if st.session_state.lang == "es":
            st.markdown("""
- **Errores de descripción:** Algunos vendedores ingresan un barrio incorrecto en el anuncio.
- **Límites de barrios difusos:** Las propiedades en los límites pueden estar registradas en un barrio pero ubicarse físicamente en otro.
- **Nombres alternativos:** Barrios con múltiples nombres (ej: "La Candelaria" vs "Las Palmas").
            """)
        else:
            st.markdown("""
- **Description Errors:** Some sellers enter an incorrect neighborhood in the listing.
- **Fuzzy Boundaries:** Properties on neighborhood borders may be registered in one area but physically located in another.
- **Alternative Names:** Neighborhoods with multiple names (e.g., "La Candelaria" vs "Las Palmas").
            """)

        st.markdown(f"**{t('disclaimer_p3')}** {t('disclaimer_p3_desc')}")

    st.markdown("---")

    kind_label = st.segmented_control(
        t('transaction_type'), [t('rental'), t('sale')], default=t('rental'))
    kind = "arr" if kind_label == t('rental') else "ven"

    # ── R² badge — updates when user switches Arriendo / Venta ──
    r2_val = MODEL_R2[kind]
    r2_color = "#0fd4c0" if r2_val >= 0.75 else "#f59e0b" if r2_val >= 0.5 else "#ef4444"
    if r2_val >= 0.80:
        r2_desc = t('excellent')
    elif r2_val >= 0.70:
        r2_desc = t('good')
    else:
        r2_desc = t('moderate')

    st.markdown(
        f"""<div style='display:inline-flex;align-items:center;gap:10px;
                background:#0e1828;border:1px solid #1e3a5f;border-radius:8px;
                padding:8px 16px;margin-bottom:16px;'>
            <span style='font-size:11px;color:#5f8ab0;text-transform:uppercase;
                         letter-spacing:.08em;'>{t('model_accuracy')} · {kind_label}</span>
            <span style='font-family:"DM Mono",monospace;font-size:20px;
                         font-weight:700;color:{r2_color};'>R² {r2_val:.4f}</span>
            <span style='font-size:11px;color:{r2_color};background:{r2_color}1a;
                         border-radius:4px;padding:2px 8px;font-weight:600;'>{r2_desc}</span>
            <span style='font-size:11px;color:#3a6080;'>{t('evaluated_test')}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown(
            f"<div class='section-header'>{t('property_data')}</div>", unsafe_allow_html=True)
        barrio = st.selectbox(t('barrio'), list_barrios)
        tipo = st.selectbox(t('transaction_type'),
                            sorted(loan["tipo"].unique()))
        area = st.slider(t('area'), 20, 500, 80)
        col_a, col_b = st.columns(2)
        with col_a:
            habitaciones = st.number_input(t('bedrooms'), 0, 10, 2)
            banos = st.number_input(t('bathrooms'), 0, 10, 2)
        with col_b:
            parqueaderos = st.number_input(t('parking'), 0, 5, 1)
            estrato_input = st.number_input(t('strata'), 1, 6, 3)

        col_c, col_d = st.columns(2)
        with col_c:
            admin_input = st.number_input(
                t('admin_fee'), 0, 2_000_000, 0, step=50_000)
        with col_d:
            metro_input = st.number_input(
                t('metro_distance'), 0.0, 10.0, 1.0, step=0.1)

        predict_btn = st.button(t('predict_btn'),
                                type="primary", use_container_width=True)

    with col_result:
        st.markdown(f"<div class='section-header'>{t('result')}</div>",
                    unsafe_allow_html=True)

        if predict_btn:
            with st.spinner(t('calculating')):
                price, ppm2, low, high, err = predict(area, habitaciones, banos, parqueaderos, barrio, tipo, kind,
                                                      estrato=estrato_input, administracion=admin_input, dist_metro_km=metro_input)

            if err:
                st.error(err)
            else:

                st.markdown(f"""
                <div class='pred-box'>
                    <div class='pred-label'>{t('estimated_price')} · {kind_label}</div>
                    <div class='pred-value'>{fmt_price(price)}</div>
                    <div class='pred-range'>{t('probable_range')}: {fmt_price(low)} — {fmt_price(high)}</div>
                </div>
                """, unsafe_allow_html=True)

                # Price per m²
                stats = get_barrio_stats(barrio)
                barrio_ppm2 = stats[f"{'arr' if kind == 'arr' else 'ven'}_ppm2"]

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='label'>{t('price_m2')}</div>
                        <div class='value'>{fmt_price(ppm2)}</div>
                        <div class='sub'>{t('your_property')}</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='label'>{t('median_barrio_m2')}</div>
                        <div class='value'>{fmt_price(barrio_ppm2)}</div>
                        <div class='sub'>{barrio}</div>
                    </div>""", unsafe_allow_html=True)

                # Feature contribution breakdown
                st.markdown(
                    f"<div class='section-header'>{t('key_factors')}</div>", unsafe_allow_html=True)

                # Approximate feature weights from what we know about the engineered features
                factors = {
                    t('area'): area / 500,
                    f"{t('median_barrio_m2')}": min((barrio_ppm2 or 0) / 30_000_000, 1.0),
                    t('bedrooms'): habitaciones / 10,
                    t('bathrooms'): banos / 10,
                    t('parking'): parqueaderos / 5,
                }
                max_w = max(factors.values()) or 1
                for name, weight in factors.items():
                    pct = int((weight / max_w) * 100)
                    st.markdown(f"""
                    <div class='feat-row'>
                        <span class='feat-name'>{name}</span>
                        <div class='feat-bar-wrap'>
                            <div class='feat-bar-fill' style='width:{pct}%'></div>
                        </div>
                        <span class='feat-val'>{pct}%</span>
                    </div>""", unsafe_allow_html=True)

                # Comparables
                st.markdown(
                    f"<div class='section-header'>{t('similar_props')}</div>", unsafe_allow_html=True)
                ref = loan if kind == "arr" else sales
                comp = ref[
                    (ref["barrio_norm"] == barrio.strip().lower()) &
                    (ref["tipo"].str.strip().str.lower() == tipo.strip().lower()) &
                    (ref["area"].between(area * 0.75, area * 1.25))
                ].head(5)

                if len(comp):
                    show_cols = [c for c in ["nombre", "tipo", "precio", "area",
                                             "habitaciones", "baños", "parqueaderos"] if c in comp.columns]
                    comp_display = comp[show_cols].copy()
                    comp_display["precio"] = comp_display["precio"].apply(
                        fmt_price)
                    comp_display = comp_display.rename(columns={
                        "nombre": t('barrio'),
                        "tipo": t('type_col'),
                        "precio": t('price_col'),
                        "area": t('area_col'),
                        "habitaciones": t('rooms_col'),
                        "baños": t('baths_col'),
                        "parqueaderos": t('parking_col'),
                    })
                    st.dataframe(
                        comp_display, use_container_width=True, hide_index=True)
                else:
                    st.caption(t('no_comparables'))

        else:
            st.markdown(f"""
            <div style='color:#3a6080; padding: 40px 20px; text-align: center; border: 1px dashed #1e3a5f; border-radius:12px;'>
                {t('fill_form')}<br><strong style='color:#5f8ab0'>{t('predict_btn')}</strong>
            </div>
            """, unsafe_allow_html=True)

            # Show neighbourhood context even before predicting
            stats = get_barrio_stats(
                barrio if "barrio" in dir() else list_barrios[0])
            st.markdown(
                f"<div class='section-header'>{t('barrio_context')}</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='label'>{t('median_rental')}</div>
                    <div class='value' style='font-size:20px'>{fmt_price(stats['arr_median'])}</div>
                    <div class='sub'>{stats['arr_count']} {t('listings')}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='label'>{t('median_sale')}</div>
                    <div class='value' style='font-size:20px'>{fmt_price(stats['ven_median'])}</div>
                    <div class='sub'>{stats['ven_count']} {t('listings')}</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: EXPLORAR DATOS
# ─────────────────────────────────────────────
elif mode == t('nav_explore'):
    st.markdown(f"## {t('explorer_title')}")

    ds_choice = st.segmented_control(
        t('dataset'), [t('rental'), t('sale')], default=t('rental'))
    df = loan.copy() if ds_choice == t('rental') else sales.copy()

    # Apply sidebar filters
    if selected_barrio != t('all'):
        df = df[df["barrio_norm"] == selected_barrio.strip().lower()]
    df = df[(df["precio"] >= min_price) & (df["precio"] <= max_price)]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (t('total_listings'),        f"{len(df):,}",
         t('filtered')),
        (t('price_median'),        fmt_price(df['precio'].median()),    "COP"),
        (t('area_median'),
         f"{df['area'].median():.0f} {t('m2')}" if 'area' in df.columns else "—", ""),
        (t('opportunities'),
         f"{df['oportunity_houses'].sum():,}" if 'oportunity_houses' in df.columns else "—", t('underpriced')),
    ]
    for col, (label, value, sub) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value' style='font-size:22px'>{value}</div>
                <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>{t('visualizations')}</div>",
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        [t('price_distribution'), t('price_vs_area'), t('by_barrio')])

    with tab1:
        if len(df):
            fig = px.histogram(
                df, x="precio", nbins=50,
                color_discrete_sequence=["#0fd4c0"],
                template="plotly_dark",
                title=f"{t('price_distribution')} — {ds_choice}",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0a0f1a",
                font=dict(family="DM Sans", color="#8aabcc"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "area" in df.columns and len(df):
            fig2 = px.scatter(
                df.sample(min(2000, len(df))),
                x="area", y="precio",
                hover_data=["nombre", "tipo"],
                color="tipo",
                template="plotly_dark",
                title=t('price_vs_area'),
                opacity=0.65,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0a0f1a",
                font=dict(family="DM Sans", color="#8aabcc"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        barrio_agg = (
            df.groupby("nombre")
            .agg(mediana=("precio", "median"), count=("precio", "count"))
            .sort_values("mediana", ascending=False)
            .head(25)
            .reset_index()
        )
        fig3 = px.bar(
            barrio_agg, x="mediana", y="nombre", orientation="h",
            color="mediana",
            color_continuous_scale="teal",
            template="plotly_dark",
            title=t('top_barrios'),
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a0f1a",
            font=dict(family="DM Sans", color="#8aabcc"),
            yaxis=dict(autorange="reversed"),
            height=560,
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"<div class='section-header'>{t('data_table')}</div>",
                unsafe_allow_html=True)
    st.dataframe(
        df.drop(columns=["barrio_norm"], errors="ignore").sample(
            min(500, len(df))) if len(df) else df,
        use_container_width=True,
        height=300,
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        t('download_csv'),
        data=csv,
        file_name=f"{ds_choice.lower()}_{st.session_state.lang}_filtrado.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# PAGE: MAPA DE OPORTUNIDADES
# ─────────────────────────────────────────────
elif mode == t('nav_map'):
    st.markdown(f"## {t('map_title')}")
    st.markdown(t('map_desc'))

    map_mode = st.segmented_control(
        t('view'), [t('sale'), t('rental')], default=t('sale'))
    df_map = sales if map_mode == t('sale') else loan

    # ── Prepare data ─────────────────────────────────────────────────────────
    opp = df_map[df_map["oportunity_houses"]].copy()
    if selected_barrio != t('all'):
        opp = opp[opp["barrio_norm"] == selected_barrio.strip().lower()]
    opp = opp[(opp["precio"] >= min_price) & (opp["precio"] <= max_price)]
    opp_with_coords = opp.reset_index(drop=True)

    # ── Build display table — pricing + opportunity signal only ─────────────
    # Compute nearest metro distance if not already in the dataframe
    if "dist_metro_km" not in opp.columns and "latitud" in opp.columns:
        metro_stations = None
        try:
            metro_stations = load_pickle(_artifact("metro_stations.pkl"))
        except Exception:
            pass
        if metro_stations:
            import numpy as _np2
            sta = _np2.array([(s[1], s[2]) for s in metro_stations])
            props = opp[["latitud", "longitud"]].values
            lat_m = props[:, 0].mean()
            scale = _np2.array([111.0, 111.0 * _np2.cos(_np2.radians(lat_m))])
            diffs = props[:, _np2.newaxis, :] - sta[_np2.newaxis, :, :]
            opp = opp.copy()
            opp["dist_metro_km"] = _np2.sqrt(
                ((diffs * scale) ** 2).sum(axis=2)).min(axis=1)

    show_cols = [c for c in ["nombre", "tipo", "precio", "cat_pred",
                             "pct_underpriced", "dist_metro_km", "url"] if c in opp.columns]
    opp_display = opp[show_cols].copy()
    opp_display["precio"] = opp_display["precio"].apply(fmt_price)
    if "cat_pred" in opp_display.columns:
        opp_display["cat_pred"] = opp_display["cat_pred"].apply(fmt_price)
    if "pct_underpriced" in opp_display.columns:
        opp_display["pct_underpriced"] = opp_display["pct_underpriced"].apply(
            lambda x: f"{x:.1f}%")
    if "dist_metro_km" in opp_display.columns:
        opp_display["dist_metro_km"] = opp_display["dist_metro_km"].apply(
            lambda x: f"{x:.2f} km" if pd.notna(x) else "—")
    if "tipo" in opp_display.columns:
        opp_display["tipo"] = opp_display["tipo"].str.title()

    opp_display = opp_display.rename(columns={
        "nombre": t('barrio_col'),
        "tipo": t('type_col'),
        "precio": t('real_price_col'),
        "cat_pred": t('model_price_col'),
        "pct_underpriced": t('discount_col'),
        "dist_metro_km": t('metro_col'),
        "url": t('link_col'),
    })

    col_cfg = {}
    if t('link_col') in opp_display.columns:
        col_cfg[t('link_col')] = st.column_config.LinkColumn(
            t('link_col'), display_text=t('open_link'), help="Abrir en metrocuadrado.com")

    has_coords = "latitud" in opp_with_coords.columns and "longitud" in opp_with_coords.columns

    # ── Neighbourhood filter (scoped to this page, above the columns) ──────────
    opp_barrios = [t('all')] + \
        sorted(opp_with_coords["nombre"].dropna().unique().tolist())
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        page_barrio = st.selectbox(
            t('filter_by_barrio'),
            opp_barrios,
            key="page_barrio_filter",
        )
    if page_barrio != t('all'):
        mask = opp_with_coords["nombre"].str.strip(
        ).str.lower() == page_barrio.strip().lower()
        opp_with_coords = opp_with_coords[mask].reset_index(drop=True)
        opp_display = opp_display[mask.values].reset_index(drop=True)

    # ── Side-by-side: table left, map right ───────────────────────────────────
    col_tbl, col_map = st.columns([1, 1], gap="medium")

    with col_tbl:
        st.markdown(
            f"<div class='section-header'>{t('properties_found')} "
            f"<span style='color:#0fd4c0'>{len(opp):,}</span> {t('found')}</div>",
            unsafe_allow_html=True,
        )
        selected_event = st.dataframe(
            opp_display,
            use_container_width=True,
            height=460,
            hide_index=True,
            column_config=col_cfg,
            on_select="rerun",
            selection_mode="single-row",
            key="opp_table",
        )

        # ── POI controls live under the table, not above the map ─────────────
        selected_rows = selected_event.selection.get(
            "rows", []) if selected_event else []
        sel_lat = sel_lon = None
        prop = None
        if selected_rows and has_coords:
            prop = opp_with_coords.iloc[selected_rows[0]]
            _lat = prop.get("latitud")
            _lon = prop.get("longitud")
            if pd.notna(_lat) and pd.notna(_lon):
                sel_lat, sel_lon = _lat, _lon

        if sel_lat:
            st.markdown(f"<div class='section-header'>{t('poi_nearby')}</div>",
                        unsafe_allow_html=True)
            poi_toggle = st.toggle(t('show_on_map'),
                                   value=True, key="poi_toggle")
            poi_cats = st.multiselect(
                t('categories'),
                options=list(POI_CATEGORIES.keys()),
                default=["🍽️ Restaurantes", "🛒 Supermercados",
                         "🌳 Parques", "🚇 Transporte"],
                key="poi_cats",
                label_visibility="collapsed",
            )
        else:
            poi_toggle = False
            poi_cats = []

        # Fetch POIs here so the result is available inside col_map
        pois = {}
        if sel_lat and poi_toggle and poi_cats:
            with st.spinner(t('searching_poi')):
                pois = fetch_pois(sel_lat, sel_lon, radius_m=800)

    with col_map:
        # ── Build map — no widgets here, purely the figure ────────────────────
        if not has_coords:
            st.caption(t('no_coords'))
        else:
            map_base = opp_with_coords.dropna(
                subset=["latitud", "longitud"]).copy()
            map_base["label"] = (
                map_base["nombre"].astype(str).str.title() + " · " +
                map_base["precio"].apply(fmt_price) + " · " +
                map_base["pct_underpriced"].apply(lambda x: f"{x:.1f}% desc")
            )

            if sel_lat:
                center_lat, center_lon, zoom = sel_lat, sel_lon, 15
            else:
                center_lat = map_base["latitud"].mean()
                center_lon = map_base["longitud"].mean()
                zoom = 12

            fig_map = go.Figure()

            # Background opportunity dots
            bg = map_base if sel_lat is None else map_base[
                ~((map_base["latitud"] == sel_lat) &
                  (map_base["longitud"] == sel_lon))
            ]
            if len(bg):
                fig_map.add_trace(go.Scattermapbox(
                    lat=bg["latitud"], lon=bg["longitud"],
                    mode="markers",
                    marker=dict(size=7, color="#1a7a6e", opacity=0.7),
                    text=bg["label"], hoverinfo="text",
                    name=t('opportunities'),
                ))

            # Selected property — bright + label
            if sel_lat:
                sel_label = (
                    f"{prop.get('tipo', '').title()} · "
                    f"{fmt_price(prop.get('precio'))} · "
                    f"{prop.get('nombre', '')} · "
                    f"{prop.get('pct_underpriced', 0):.1f}% desc"
                )
                fig_map.add_trace(go.Scattermapbox(
                    lat=[sel_lat], lon=[sel_lon],
                    mode="markers+text",
                    marker=dict(size=20, color="#0fd4c0", opacity=1.0),
                    text=[prop.get("nombre", "").title()],
                    textposition="top center",
                    textfont=dict(size=12, color="#0fd4c0"),
                    hovertext=[sel_label], hoverinfo="text",
                    name="Seleccionada",
                ))

                # POI traces — data already fetched in col_tbl
                for cat in poi_cats:
                    if cat in pois and len(pois[cat]):
                        df_poi = pois[cat].copy()
                        # Build rich hover: name + rating stars + cuisine

                        def _poi_label(row):
                            parts = [row["name"]]
                            if pd.notna(row.get("rating")) and row.get("rating"):
                                stars = "★" * \
                                    int(round(row["rating"])) + "☆" * \
                                    (5 - int(round(row["rating"])))
                                parts.append(f"{stars} ({row['rating']:.1f})")
                            if row.get("cuisine"):
                                parts.append(
                                    row["cuisine"].replace(";", ", ").title())
                            if row.get("opening"):
                                parts.append(f"🕐 {row['opening'][:30]}")
                            return "<br>".join(parts)
                        df_poi["hover"] = df_poi.apply(_poi_label, axis=1)
                        # Size by rating if available, else uniform
                        has_rating = df_poi["rating"].notna() & (
                            df_poi["rating"] > 0)
                        sizes = (df_poi["rating"].fillna(3) /
                                 5 * 10 + 6).clip(6, 16).tolist()
                        fig_map.add_trace(go.Scattermapbox(
                            lat=df_poi["lat"], lon=df_poi["lon"],
                            mode="markers",
                            marker=dict(
                                size=sizes,
                                color=POI_COLORS.get(cat, "#ffffff"),
                                opacity=0.9,
                            ),
                            text=df_poi["hover"], hoverinfo="text",
                            name=cat,
                        ))

            fig_map.update_layout(
                mapbox=dict(
                    style="carto-darkmatter",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom,
                ),
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)",
                height=560,
                legend=dict(
                    bgcolor="rgba(10,15,26,0.85)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(color="#8aabcc", size=11),
                    x=0.01, y=0.99,
                ),
                showlegend=sel_lat is not None,
            )

            if sel_lat:
                st.caption(
                    f"📍 **{prop.get('tipo', '').title()} · "
                    f"{fmt_price(prop.get('precio'))} · "
                    f"{prop.get('nombre', '')}** — "
                    f"{prop.get('pct_underpriced', 0):.1f}% {t('discount_col').lower()}"
                )
            else:
                st.caption(t('select_row'))
            st.plotly_chart(fig_map, use_container_width=True)

    # ── Scatter: real vs predicted ──
    st.markdown(f"<div class='section-header'>{t('real_vs_estimated')}</div>",
                unsafe_allow_html=True)
    scatter_df = df_map[["precio", "cat_pred", "barrio_norm", "tipo",
                         "oportunity_houses"]].dropna().sample(min(1500, len(df_map)))
    fig_sc = px.scatter(
        scatter_df,
        x="cat_pred", y="precio",
        color="oportunity_houses",
        color_discrete_map={True: "#0fd4c0", False: "#1e3a5f"},
        hover_data=["barrio_norm", "tipo"],
        labels={"cat_pred": t('model_price_col'), "precio": t('real_price_col'),
                "oportunity_houses": t('opportunities')},
        template="plotly_dark",
        title=f"{t('real_vs_estimated')} — teal = {t('opportunities').lower()}",
        opacity=0.7,
    )
    max_val = scatter_df[["precio", "cat_pred"]].max().max()
    fig_sc.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color="#ffffff", width=1, dash="dot"),
        name=t('fair_price'), showlegend=True,
    ))
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0f1a",
        font=dict(family="DM Sans", color="#8aabcc"),
        height=420,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    csv2 = opp.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"{t('download_opportunities')} {map_mode.lower()}",
        data=csv2,
        file_name=f"oportunidades_{map_mode.lower()}_{st.session_state.lang}.csv",
        mime="text/csv",
    )
