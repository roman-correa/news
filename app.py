"""
Compra/Venta Medellín — Real Estate Price Predictor
Author: Roman Alejandro Correa
"""
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
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
# StackedEnsemble — must be defined here so pickle can deserialise it
# (pickle looks up the class in the module where it's being loaded)
# ─────────────────────────────────────────────


class StackedEnsemble:
    """XGBoost + LightGBM + CatBoost → Ridge meta-learner."""

    def __init__(self, xgb_p=None, lgb_p=None, cat_p=None):
        self.xgb_p = xgb_p or {}
        self.lgb_p = lgb_p or {}
        self.cat_p = cat_p or {}
        self.base_ = []
        self.meta_ = None

    def predict(self, X) -> np.ndarray:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
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
        return "desconocida"


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
        return f'Barrio "{barrio}" no está en la lista.'
    tipos_lc = [t.strip().lower() for t in reference_df["tipo"].unique()]
    if tipo.strip().lower() not in tipos_lc:
        return f'Tipo "{tipo}" no está en la lista.'
    return None


def predict(area, habitaciones, banos, parqueaderos, barrio, tipo, kind="arr", **kwargs) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """Return (prediction, price_per_m2, error_string)."""
    ref_df = loan if kind == "arr" else sales
    err = _validate(barrio, tipo, ref_df)
    if err:
        return None, None, err

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
        return None, None, None, None, f"Faltan columnas: {missing}"

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
    "🍽️ Restaurantes":      '[amenity~"restaurant|cafe|fast_food"]',
    "🛒 Supermercados":      '[shop~"supermarket|convenience"]',
    "🏬 Centro comercial":   '[shop="mall"]',
    "🏥 Salud":              '[amenity~"hospital|clinic|pharmacy|dentist"]',
    "🏫 Educación":          '[amenity~"school|university|college|library"]',
    "🚇 Transporte":         '[amenity~"bus_station|taxi"][public_transport~"station|stop_area"]',
    "💪 Gimnasios":          '[leisure~"fitness_centre|sports_centre"]',
    "🌳 Parques":            '[leisure="park"]',
    "🏦 Bancos/ATM":         '[amenity~"bank|atm"]',
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
    st.markdown("### 🏙️ Medellín RE")
    st.markdown("<div class='section-header'>Navegación</div>",
                unsafe_allow_html=True)
    mode = st.radio(
        "Sección",
        ["🔮 Predictor", "📊 Explorar datos", "🗺️ Mapa de oportunidades"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='section-header'>Filtros globales</div>",
                unsafe_allow_html=True)
    selected_barrio = st.selectbox("Barrio", ["Todos"] + sorted(list_barrios))
    min_price = st.number_input(
        "Precio mínimo (COP)", value=0, step=1_000_000, format="%d")
    max_price = st.number_input(
        "Precio máximo (COP)", value=2_000_000_000, step=1_000_000, format="%d")

    st.markdown("---")
    st.markdown("<div class='section-header'>Estado del modelo</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"""<div style='font-size:12px;color:#5f8ab0;line-height:2;'>
            📅 <b style='color:#8aabcc'>Datos actualizados</b><br>
            <span style='font-family:"DM Mono",monospace;color:#0fd4c0'>{DATA_DATE}</span><br><br>
            📐 <b style='color:#8aabcc'>R² Arriendo</b><br>
            <span style='font-family:"DM Mono",monospace;color:#0fd4c0'>{MODEL_R2.get("arr", float("nan")):.4f}</span><br><br>
            📐 <b style='color:#8aabcc'>R² Venta</b><br>
            <span style='font-family:"DM Mono",monospace;color:#0fd4c0'>{MODEL_R2.get("ven", float("nan")):.4f}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Por Roman Alejandro Correa")


# ─────────────────────────────────────────────
# PAGE: PREDICTOR
# ─────────────────────────────────────────────
if mode == "🔮 Predictor":
    st.markdown("## Predictor de precios")
    st.markdown(
        "Estima el precio de arriendo o venta de una propiedad en Medellín.")

    kind_label = st.segmented_control(
        "Tipo de transacción", ["Arriendo", "Venta"], default="Arriendo")
    kind = "arr" if kind_label == "Arriendo" else "ven"

    # ── R² badge — updates when user switches Arriendo / Venta ──
    r2_val = MODEL_R2[kind]
    r2_color = "#0fd4c0" if r2_val >= 0.75 else "#f59e0b" if r2_val >= 0.5 else "#ef4444"
    r2_desc = "Excelente" if r2_val >= 0.80 else "Bueno" if r2_val >= 0.70 else "Moderado"
    st.markdown(
        f"""<div style='display:inline-flex;align-items:center;gap:10px;
                background:#0e1828;border:1px solid #1e3a5f;border-radius:8px;
                padding:8px 16px;margin-bottom:16px;'>
            <span style='font-size:11px;color:#5f8ab0;text-transform:uppercase;
                         letter-spacing:.08em;'>Precisión del modelo · {kind_label}</span>
            <span style='font-family:"DM Mono",monospace;font-size:20px;
                         font-weight:700;color:{r2_color};'>R² {r2_val:.4f}</span>
            <span style='font-size:11px;color:{r2_color};background:{r2_color}1a;
                         border-radius:4px;padding:2px 8px;font-weight:600;'>{r2_desc}</span>
            <span style='font-size:11px;color:#3a6080;'>· evaluado en conjunto de prueba (log-space)</span>
        </div>""",
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown(
            "<div class='section-header'>Datos de la propiedad</div>", unsafe_allow_html=True)
        barrio = st.selectbox("Barrio", list_barrios)
        tipo = st.selectbox("Tipo", sorted(loan["tipo"].unique()))
        area = st.slider("Área (m²)", 20, 500, 80)
        col_a, col_b = st.columns(2)
        with col_a:
            habitaciones = st.number_input("Habitaciones", 0, 10, 2)
            banos = st.number_input("Baños", 0, 10, 2)
        with col_b:
            parqueaderos = st.number_input("Parqueaderos", 0, 5, 1)
            estrato_input = st.number_input("Estrato", 1, 6, 3)

        col_c, col_d = st.columns(2)
        with col_c:
            admin_input = st.number_input(
                "Admon. mensual (COP)", 0, 2_000_000, 0, step=50_000)
        with col_d:
            metro_input = st.number_input(
                "Dist. metro aprox. (km)", 0.0, 10.0, 1.0, step=0.1)

        predict_btn = st.button("Predecir precio →",
                                type="primary", use_container_width=True)

    with col_result:
        st.markdown("<div class='section-header'>Resultado</div>",
                    unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Calculando..."):
                price, ppm2, low, high, err = predict(area, habitaciones, banos, parqueaderos, barrio, tipo, kind,
                                                      estrato=estrato_input, administracion=admin_input, dist_metro_km=metro_input)

            if err:
                st.error(err)
            else:

                st.markdown(f"""
                <div class='pred-box'>
                    <div class='pred-label'>Precio estimado · {kind_label}</div>
                    <div class='pred-value'>{fmt_price(price)}</div>
                    <div class='pred-range'>Rango probable: {fmt_price(low)} — {fmt_price(high)}</div>
                </div>
                """, unsafe_allow_html=True)

                # Price per m²
                stats = get_barrio_stats(barrio)
                barrio_ppm2 = stats[f"{'arr' if kind == 'arr' else 'ven'}_ppm2"]

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='label'>Precio / m²</div>
                        <div class='value'>{fmt_price(ppm2)}</div>
                        <div class='sub'>Tu propiedad</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='label'>Mediana barrio / m²</div>
                        <div class='value'>{fmt_price(barrio_ppm2)}</div>
                        <div class='sub'>{barrio}</div>
                    </div>""", unsafe_allow_html=True)

                # Feature contribution breakdown
                st.markdown(
                    "<div class='section-header'>Factores clave del precio</div>", unsafe_allow_html=True)

                # Approximate feature weights from what we know about the engineered features
                factors = {
                    "Área (m²)": area / 500,
                    "Precio/m² del barrio": min((barrio_ppm2 or 0) / 30_000_000, 1.0),
                    "Habitaciones": habitaciones / 10,
                    "Baños": banos / 10,
                    "Parqueaderos": parqueaderos / 5,
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
                    "<div class='section-header'>Propiedades similares en el barrio</div>", unsafe_allow_html=True)
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
                    st.dataframe(
                        comp_display, use_container_width=True, hide_index=True)
                else:
                    st.caption(
                        "No hay comparables con filtros exactos en este barrio.")

        else:
            st.markdown("""
            <div style='color:#3a6080; padding: 40px 20px; text-align: center; border: 1px dashed #1e3a5f; border-radius:12px;'>
                Completa el formulario y presiona<br><strong style='color:#5f8ab0'>Predecir precio →</strong>
            </div>
            """, unsafe_allow_html=True)

            # Show neighbourhood context even before predicting
            stats = get_barrio_stats(
                barrio if "barrio" in dir() else list_barrios[0])
            st.markdown(
                "<div class='section-header'>Contexto del barrio seleccionado</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='label'>Arriendo — mediana</div>
                    <div class='value' style='font-size:20px'>{fmt_price(stats['arr_median'])}</div>
                    <div class='sub'>{stats['arr_count']} listings</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='label'>Venta — mediana</div>
                    <div class='value' style='font-size:20px'>{fmt_price(stats['ven_median'])}</div>
                    <div class='sub'>{stats['ven_count']} listings</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: EXPLORAR DATOS
# ─────────────────────────────────────────────
elif mode == "📊 Explorar datos":
    st.markdown("## Explorador de datos")

    ds_choice = st.segmented_control(
        "Dataset", ["Arriendo", "Venta"], default="Arriendo")
    df = loan.copy() if ds_choice == "Arriendo" else sales.copy()

    # Apply sidebar filters
    if selected_barrio != "Todos":
        df = df[df["barrio_norm"] == selected_barrio.strip().lower()]
    df = df[(df["precio"] >= min_price) & (df["precio"] <= max_price)]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total listings",        f"{len(df):,}",
         "filtrados"),
        ("Precio mediana",        fmt_price(df['precio'].median()),    "COP"),
        ("Área mediana",
         f"{df['area'].median():.0f} m²" if 'area' in df.columns else "—", ""),
        ("Oportunidades",
         f"{df['oportunity_houses'].sum():,}" if 'oportunity_houses' in df.columns else "—", ">20% subvaloradas"),
    ]
    for col, (label, value, sub) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value' style='font-size:22px'>{value}</div>
                <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Visualizaciones</div>",
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["Distribución de precios", "Precio vs Área", "Por barrio"])

    with tab1:
        if len(df):
            fig = px.histogram(
                df, x="precio", nbins=50,
                color_discrete_sequence=["#0fd4c0"],
                template="plotly_dark",
                title=f"Distribución de precios — {ds_choice}",
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
                title="Precio vs Área",
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
            title="Top 25 barrios por precio mediano",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a0f1a",
            font=dict(family="DM Sans", color="#8aabcc"),
            yaxis=dict(autorange="reversed"),
            height=560,
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Tabla de datos</div>",
                unsafe_allow_html=True)
    st.dataframe(
        df.drop(columns=["barrio_norm"], errors="ignore").sample(
            min(500, len(df))) if len(df) else df,
        use_container_width=True,
        height=300,
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Descargar CSV filtrado",
        data=csv,
        file_name=f"{ds_choice.lower()}_filtrado.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# PAGE: MAPA DE OPORTUNIDADES
# ─────────────────────────────────────────────
elif mode == "🗺️ Mapa de oportunidades":
    st.markdown("## Mapa de oportunidades")
    st.markdown(
        "Barrios y propiedades donde el precio de mercado está por debajo del valor predicho por el modelo.")

    map_mode = st.segmented_control(
        "Ver", ["Venta", "Arriendo"], default="Venta")
    df_map = sales if map_mode == "Venta" else loan

    # ── Prepare data ─────────────────────────────────────────────────────────
    opp = df_map[df_map["oportunity_houses"]].copy()
    if selected_barrio != "Todos":
        opp = opp[opp["barrio_norm"] == selected_barrio.strip().lower()]
    opp = opp[(opp["precio"] >= min_price) & (opp["precio"] <= max_price)]
    opp_with_coords = opp.reset_index(drop=True)

    show_cols = [c for c in ["nombre", "tipo", "precio", "cat_pred", "pct_underpriced",
                             "area", "habitaciones", "baños", "parqueaderos", "url"] if c in opp.columns]
    opp_display = opp[show_cols].copy()
    if "precio" in opp_display.columns:
        opp_display["precio"] = opp_display["precio"].apply(fmt_price)
    if "cat_pred" in opp_display.columns:
        opp_display["cat_pred"] = opp_display["cat_pred"].apply(fmt_price)
    if "pct_underpriced" in opp_display.columns:
        opp_display["pct_underpriced"] = opp_display["pct_underpriced"].apply(
            lambda x: f"{x:.1f}%")
    opp_display = opp_display.rename(columns={
        "nombre": "Barrio", "tipo": "Tipo", "precio": "Precio real",
        "cat_pred": "Precio modelo", "pct_underpriced": "Descuento",
        "area": "m²", "habitaciones": "Hab", "baños": "Baños",
        "parqueaderos": "Parq", "url": "Link",
    })
    if "Tipo" in opp_display.columns:
        opp_display["Tipo"] = opp_display["Tipo"].str.title()
    col_cfg = {}
    if "Link" in opp_display.columns:
        col_cfg["Link"] = st.column_config.LinkColumn(
            "Link", display_text="Ver →", help="Abrir en metrocuadrado.com")

    has_coords = "latitud" in opp_with_coords.columns and "longitud" in opp_with_coords.columns

    # ── Neighbourhood filter (scoped to this page, above the columns) ──────────
    opp_barrios = ["Todos"] + \
        sorted(opp_with_coords["nombre"].dropna().unique().tolist())
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        page_barrio = st.selectbox(
            "Filtrar por barrio",
            opp_barrios,
            key="page_barrio_filter",
        )
    if page_barrio != "Todos":
        mask = opp_with_coords["nombre"].str.strip(
        ).str.lower() == page_barrio.strip().lower()
        opp_with_coords = opp_with_coords[mask].reset_index(drop=True)
        opp_display = opp_display[mask.values].reset_index(drop=True)

    # ── Side-by-side: table left, map right ───────────────────────────────────
    col_tbl, col_map = st.columns([1, 1], gap="medium")

    with col_tbl:
        st.markdown(
            f"<div class='section-header'>Propiedades oportunidad "
            f"<span style='color:#0fd4c0'>{len(opp):,}</span> encontradas</div>",
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
            st.markdown("<div class='section-header'>Puntos de interés cercanos</div>",
                        unsafe_allow_html=True)
            poi_toggle = st.toggle("Mostrar en el mapa",
                                   value=True, key="poi_toggle")
            poi_cats = st.multiselect(
                "Categorías",
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
            with st.spinner("Buscando puntos de interés..."):
                pois = fetch_pois(sel_lat, sel_lon, radius_m=800)

    with col_map:
        # ── Build map — no widgets here, purely the figure ────────────────────
        if not has_coords:
            st.caption("Sin coordenadas disponibles.")
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
                    name="Oportunidades",
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
                        df_poi = pois[cat]
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
                    f"{prop.get('pct_underpriced', 0):.1f}% bajo modelo"
                )
            else:
                st.caption(
                    "Selecciona una fila para ver la propiedad y sus POIs.")
            st.plotly_chart(fig_map, use_container_width=True)

    # ── Scatter: real vs predicted ──
    st.markdown("<div class='section-header'>Precio real vs. precio estimado por el modelo</div>",
                unsafe_allow_html=True)
    scatter_df = df_map[["precio", "cat_pred", "barrio_norm", "tipo",
                         "oportunity_houses"]].dropna().sample(min(1500, len(df_map)))
    fig_sc = px.scatter(
        scatter_df,
        x="cat_pred", y="precio",
        color="oportunity_houses",
        color_discrete_map={True: "#0fd4c0", False: "#1e3a5f"},
        hover_data=["barrio_norm", "tipo"],
        labels={"cat_pred": "Precio modelo", "precio": "Precio real",
                "oportunity_houses": "Oportunidad"},
        template="plotly_dark",
        title="Precio real vs estimado — teal = oportunidad",
        opacity=0.7,
    )
    max_val = scatter_df[["precio", "cat_pred"]].max().max()
    fig_sc.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color="#ffffff", width=1, dash="dot"),
        name="Precio justo", showlegend=True,
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
        f"⬇ Descargar oportunidades {map_mode.lower()}",
        data=csv2,
        file_name=f"oportunidades_{map_mode.lower()}.csv",
        mime="text/csv",
    )
