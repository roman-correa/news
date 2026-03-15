"""
Medellín Real Estate — Training Pipeline  v2
============================================
Scrapes metrocuadrado.com + Fincaraíz, cleans data, engineers features
(including estrato, metro distance, amenities), trains a stacked ensemble
(XGBoost + LightGBM + CatBoost → Ridge meta-learner), produces quantile
predictions for confidence intervals, and logs everything to MLflow.

Usage:
    python train.py                          # scrape + train both
    python train.py --skip-scrape            # use existing raw CSVs
    python train.py --mode arriendo          # train arriendo only
    python train.py --mode venta             # train venta only
    python train.py --skip-scrape --fast     # 20 Optuna trials (dev)

Outputs  →  ./artifacts/
    preprocessor_arr.pkl / preprocessor_ven.pkl
    stack_arr.pkl        / stack_ven.pkl          (stacked ensemble)
    q10_arr.pkl  q90_arr.pkl / q10_ven.pkl  q90_ven.pkl  (quantile models)
    best_features_arr.pkl / best_features_ven.pkl
    barrio_te_arr.pkl    / barrio_te_ven.pkl
    price_per_m2_*.pkl   price_per_space_*.pkl   price_per_parking_*.pkl
    metro_stations.pkl
    list_barrios.pkl
    model_r2.pkl
    arr_mede_final.csv   / ven_mede_final.csv
"""

from __future__ import annotations

import argparse
import pickle
import time
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import mlflow
import numpy as np
import optuna
import pandas as pd
import requests
from geopy.distance import geodesic
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MEDELLIN_BOUNDS = dict(lat_min=6.175, lat_max=6.28,
                       lon_min=-75.62, lon_max=-75.55)

# Metro de Medellín + Tranvía + Cables stations (name, lat, lon)
METRO_STATIONS = [
    ("Niquía",          6.338056, -75.536389),
    ("Bello",           6.326667, -75.536944),
    ("Madera",          6.315278, -75.540556),
    ("Acevedo",         6.298889, -75.545278),
    ("Tricentenario",   6.291111, -75.546944),
    ("Caribe",          6.279167, -75.551389),
    ("Universidad",     6.267778, -75.567222),
    ("Hospital",        6.261389, -75.573333),
    ("Prado",           6.257222, -75.572778),
    ("Parque Berrío",   6.251389, -75.568611),
    ("San Antonio",     6.247500, -75.569167),
    ("Alpujarra",       6.245278, -75.572500),
    ("Exposiciones",    6.241667, -75.573889),
    ("Industriales",    6.236111, -75.574444),
    ("Poblado",         6.210556, -75.574167),
    ("Aguacatala",      6.199722, -75.573889),
    ("Ayurá",           6.185556, -75.574722),
    ("Envigado",        6.175278, -75.583889),
    ("San Javier",      6.247222, -75.598333),
    ("Floresta",        6.248056, -75.602222),
    ("Santa Lucía",     6.248889, -75.609167),
    ("El Pedregal",     6.249722, -75.614722),
    ("La Mota",         6.250000, -75.620278),
    ("Aranjuez",        6.281667, -75.555556),
    ("Andalucía",       6.286944, -75.550278),
    ("Villa Sierra",    6.294167, -75.543611),
    ("Santo Domingo",   6.302222, -75.538889),
]

NUM_ATTRIBS = [
    # Core
    "area", "habitaciones", "baños", "parqueaderos", "espacios",
    # Area interactions
    "axe", "axh", "axa", "parq2", "garaje_bin",
    # Key Colombian predictor
    "estrato",
    # Admin fee
    "administracion",
    # Proximity
    "dist_metro_km",
    # Neighbourhood market signals
    "ppmc", "pppz", "pppp", "new_index", "barrio_count",
    "pppp/pppz", "pppp/ppmc", "pppz/ppmc",
    # Smoothed target encoding (train-only)
    "barrio_te",
]
CAT_ATTRIBS = ["tipo"]
MLFLOW_EXPERIMENT = "medellin_re"


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCRAPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mc_headers() -> dict:
    return {
        "accept": "*/*",
        "content-type": "application/json",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-api-key": "P1MfFHfQMOtL16Zpg36NcntJYCLFm8FqFfudnavl",
    }


def scrape_metrocuadrado(operacion: str = "arriendo", ciudad: str = "medellin",
                         num_paginas: int = 201, delay: float = 0.3) -> pd.DataFrame:
    """Scrape metrocuadrado — includes estrato and administración."""
    records = []
    print(f"  [metrocuadrado] {operacion} — {num_paginas} pages...")

    for page in range(num_paginas):
        params = {"size": "50", "from": str(page * 50),
                  "realEstateBusinessList": operacion, "city": ciudad}
        try:
            r = requests.get("https://www.metrocuadrado.com/rest-search/search",
                             params=params, headers=_mc_headers(), timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    Page {page}: {e} — skipping")
            continue

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            def g(k, d=None):
                try:
                    return item.get(k)
                except:
                    return d

            precio = g("mvalorarriendo") if operacion == "arriendo" else g(
                "mvalorventa")
            loc = g("localizacion") or {}
            tipo_obj = g("mtipoinmueble") or {}
            link = g("link", "") or ""

            records.append({
                "barrio":         g("mnombrecomunbarrio"),
                "precio":         precio,
                "area":           g("marea"),
                "habitaciones":   g("mnrocuartos"),
                "baños":          g("mnrobanos"),
                "parqueaderos":   g("mnrogarajes"),
                "estrato":        g("mextrato"),
                "administracion": g("madministracion"),
                "tipo":           tipo_obj.get("nombre"),
                "latitud":        loc.get("lat"),
                "longitud":       loc.get("lon"),
                "url":            "https://www.metrocuadrado.com" + link if link else None,
                "source":         "metrocuadrado",
            })

        if (page + 1) % 20 == 0:
            print(f"    {page+1}/{num_paginas} pages, {len(records)} records")
        time.sleep(delay)

    df = pd.DataFrame(records)
    df = df[df["tipo"].str.contains("Apartamento|Casa", na=False, case=False)]
    print(f"  [metrocuadrado] {len(df)} listings")
    return df


def scrape_fincaraiz(operacion: str = "arriendo", ciudad: str = "medellin",
                     num_paginas: int = 100, delay: float = 0.5) -> pd.DataFrame:
    """
    Fincaraíz does not expose a public JSON API.
    Returns an empty DataFrame so scrape_all() proceeds with metrocuadrado only.
    To add a second source in the future, implement an HTML scraper here using
    requests + BeautifulSoup against https://www.fincaraiz.com.co/
    """
    print("  [fincaraiz] No public API available — skipping (metrocuadrado only)")
    return pd.DataFrame()


def scrape_all(operacion: str) -> pd.DataFrame:
    mc = scrape_metrocuadrado(operacion)
    fr = scrape_fincaraiz(operacion)
    out = pd.concat([mc, fr], ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["area", "precio", "latitud", "longitud"])
    print(f"  Combined: {before} → {len(out)} rows after dedup")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(s) -> str:
    s = str(s).strip().lower()
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")


def clean_nombre(s) -> str:
    s = str(s).strip()
    for pfx in ["Área de Expansión", "Area de Expansion"]:
        s = s.replace(pfx, "").strip(" -–—")
    return s


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    b = MEDELLIN_BOUNDS
    before = len(df)
    df = df.dropna(subset=["latitud", "longitud"])
    df = df[
        (df["latitud"] >= b["lat_min"]) & (df["latitud"] <= b["lat_max"]) &
        (df["longitud"] >= b["lon_min"]) & (df["longitud"] <= b["lon_max"])
    ].copy()
    print(f"    fix_coordinates: dropped {before - len(df)} rows")
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame, geo: gpd.GeoDataFrame, operacion: str) -> pd.DataFrame:
    print(f"  Cleaning {operacion}: {len(df)} rows...")

    df = df.dropna(subset=["precio", "area", "habitaciones", "baños"])
    df = df[(df["precio"] > 0) & (df["area"] > 0)]

    for col in ["habitaciones", "baños", "parqueaderos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            0).apply(np.floor).astype(int)

    # Estrato 1–6 only; fill with barrio median after sjoin
    df["estrato"] = pd.to_numeric(df.get("estrato"), errors="coerce")
    df.loc[~df["estrato"].between(1, 6), "estrato"] = np.nan

    # Admin fee: cap at 5M COP, fill missing with 0
    df["administracion"] = pd.to_numeric(
        df.get("administracion"), errors="coerce").clip(0, 5_000_000).fillna(0)

    df["tipo"] = df["tipo"].astype(str).str.strip().str.lower()
    df = df[df["tipo"].str.contains("apartamento|casa", na=False)]

    df = fix_coordinates(df)

    # Spatial join → official barrio name from shapefile
    gdf_pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]), crs="EPSG:4326"
    )
    joined = gdf_pts.sjoin(geo.to_crs("EPSG:4326"), how="left")
    joined = joined[~joined.index.duplicated(
        keep="first")].reset_index(drop=True)
    joined.columns = joined.columns.str.lower()
    df = pd.DataFrame(joined).reset_index(drop=True)
    df["nombre"] = df["nombre"].astype(str).apply(clean_nombre)
    df = df[~df["nombre"].str.lower().isin(["nan", "none", ""])]

    # Fill missing estrato with barrio median, then global median
    e_med = df.groupby("nombre")["estrato"].median()
    df["estrato"] = df["estrato"].fillna(df["nombre"].map(e_med))
    df["estrato"] = df["estrato"].fillna(df["estrato"].median()).clip(1, 6)

    # Outlier removal — 3× IQR keeps luxury
    price_floor = 500_000 if operacion == "arriendo" else 50_000_000
    df = df[df["precio"] >= price_floor]
    Q1, Q3 = df["precio"].quantile(0.25), df["precio"].quantile(0.75)
    df = df[df["precio"] <= Q3 + 3.0 * (Q3 - Q1)]
    df = df[df["area"] <= df["area"].quantile(0.99)]

    keep = ["tipo", "precio", "area", "habitaciones", "baños", "parqueaderos",
            "estrato", "administracion", "nombre", "latitud", "longitud", "url", "source"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["nombre"]).reset_index(drop=True)

    print(f"  After cleaning: {len(df)} rows | "
          f"precio {df['precio'].min():,.0f}–{df['precio'].max():,.0f} | "
          f"estrato coverage {df['estrato'].notna().mean()*100:.0f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_metro_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised nearest-station distance in km."""
    print("  Computing metro distances...")
    station_coords = np.array([(slat, slon)
                              for _, slat, slon in METRO_STATIONS])
    props = df[["latitud", "longitud"]].values
    lat_m = props[:, 0].mean()
    scale = np.array([111.0, 111.0 * np.cos(np.radians(lat_m))])
    diffs = props[:, np.newaxis, :] - station_coords[np.newaxis, :, :]
    dist = np.sqrt(((diffs * scale) ** 2).sum(axis=2)).min(axis=1)
    df = df.copy()
    df["dist_metro_km"] = dist
    return df


def price_aggregates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    pt = df.groupby("nombre")["precio"].sum().astype(float)
    at = df.groupby("nombre")["area"].sum().replace(0, np.nan).astype(float)
    esp = df.groupby("nombre")["espacios"].sum().astype(float)
    pq = (df.groupby("nombre")["parqueaderos"].sum() + 1).astype(float)
    return (pt / at).dropna(), (pt / esp).dropna(), (pt / pq).dropna()


def engineer_features(df: pd.DataFrame,
                      ppmc: pd.Series, pppz: pd.Series, pppp: pd.Series) -> pd.DataFrame:
    """All features except barrio_te — computed train-only in prep()."""
    df = df.copy()
    df["espacios"] = df["habitaciones"] + df["parqueaderos"] + df["baños"]
    df["axe"] = np.where(df["espacios"] == 0, df["area"],
                         df["area"] / df["espacios"])
    df["axh"] = np.where(df["habitaciones"] == 0, df["area"],
                         df["area"] / df["habitaciones"])
    df["axa"] = df["area"] ** 2
    df["parq2"] = df["parqueaderos"] ** 2
    df["garaje_bin"] = (df["parqueaderos"] > 0).astype(int)
    df["ppmc"] = df["nombre"].map(ppmc)
    df["pppz"] = df["nombre"].map(pppz)
    df["pppp"] = df["nombre"].map(pppp)
    df["new_index"] = df["ppmc"] / ppmc.max() * 100

    def sr(a, b):
        return (a / b).replace([np.inf, -np.inf], np.nan).fillna(1).astype(float)

    df["pppp/pppz"] = sr(df["pppp"], df["pppz"])
    df["pppp/ppmc"] = sr(df["pppp"], df["ppmc"])
    df["pppz/ppmc"] = sr(df["pppz"], df["ppmc"])
    df["barrio_count"] = df["nombre"].map(
        df.groupby("nombre")["precio"].count())
    df["barrio_te"] = np.nan

    df = df.dropna(subset=["ppmc", "pppz", "pppp", "pppz/ppmc"])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]),
         NUM_ATTRIBS),
        ("cat", Pipeline(
            [("ohe",   OneHotEncoder(handle_unknown="ignore"))]), CAT_ATTRIBS),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_features(X: pd.DataFrame, y: pd.Series, max_features: int = 22) -> list[str]:
    probe = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                         subsample=0.8, random_state=42, verbosity=0)
    probe.fit(X, y)
    top = pd.Series(probe.feature_importances_, index=X.columns).nlargest(
        max_features).index.tolist()
    cv_r2 = cross_val_score(probe, X[top], y, cv=5, scoring="r2").mean()
    print(f"  Feature selection: {len(top)} features | CV R²: {cv_r2:.4f}")
    print(f"  {top}")
    return top


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTUNA TUNING
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_obj(trial, X, y):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500),
        learning_rate=trial.suggest_float(
            "learning_rate", 0.005, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 3, 20),
        subsample=trial.suggest_float("subsample", 0.6, 0.95),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.95),
        gamma=trial.suggest_float("gamma", 0.0, 0.5),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
        random_state=42, verbosity=0, n_jobs=-1,
    )
    return cross_val_score(XGBRegressor(**p), _safe_X(X), y, cv=5, scoring="r2").mean()


def _lgb_obj(trial, X, y):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500),
        learning_rate=trial.suggest_float(
            "learning_rate", 0.005, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 7),
        num_leaves=trial.suggest_int("num_leaves", 15, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
        subsample=trial.suggest_float("subsample", 0.6, 0.95),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.95),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
        random_state=42, verbose=-1, n_jobs=-1,
    )
    return cross_val_score(LGBMRegressor(**p), _safe_X(X), y, cv=5, scoring="r2").mean()


def _safe_X(X: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/nan with column medians — applied before every model fit."""
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median())


def _cat_obj(trial, X, y):
    p = dict(
        iterations=trial.suggest_int("iterations", 300, 1000),
        learning_rate=trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True),
        depth=trial.suggest_int("depth", 4, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        random_seed=42, verbose=0,
    )
    return cross_val_score(CatBoostRegressor(**p), _safe_X(X), y, cv=5, scoring="r2").mean()


def tune(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
    best = {}
    for name, obj in [("xgb", _xgb_obj), ("lgb", _lgb_obj), ("cat", _cat_obj)]:
        print(f"  Tuning {name} ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        X_safe = _safe_X(X)
        study.optimize(lambda t: obj(t, X_safe, y), n_trials=n_trials)
        best[name] = study.best_params
        print(f"    {name} best CV R²: {study.best_value:.4f}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 7. STACKED ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

class StackedEnsemble:
    """
    Level-0 : XGBoost + LightGBM + CatBoost  (OOF predictions, 5-fold CV)
    Level-1 : Ridge meta-learner
    """

    def __init__(self, xgb_p: dict, lgb_p: dict, cat_p: dict):
        self.xgb_p, self.lgb_p, self.cat_p = xgb_p, lgb_p, cat_p
        self.base_: list = []
        self.meta_: Ridge | None = None

    def _factories(self):
        return [
            lambda: XGBRegressor(
                **self.xgb_p, random_state=42, verbosity=0, n_jobs=-1),
            lambda: LGBMRegressor(
                **self.lgb_p, random_state=42, verbose=-1,  n_jobs=-1),
            lambda: CatBoostRegressor(**self.cat_p, random_seed=42, verbose=0),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> "StackedEnsemble":
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        oof = np.zeros((len(X), 3))

        print("  Training stacked ensemble (5-fold OOF)...")
        for mi, factory in enumerate(self._factories()):
            name = ["XGB", "LGB", "CAT"][mi]
            fold_r2 = []
            for tr_i, va_i in kf.split(X):
                m = factory()
                m.fit(_safe_X(X.iloc[tr_i]), y.iloc[tr_i])
                oof[va_i, mi] = m.predict(_safe_X(X.iloc[va_i]))
                fold_r2.append(r2_score(y.iloc[va_i], oof[va_i, mi]))
            print(
                f"    {name} OOF R²: {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
            m_full = factory()
            m_full.fit(_safe_X(X), y)
            self.base_.append(m_full)

        self.meta_ = Ridge(alpha=1.0)
        self.meta_.fit(oof, y)
        print(
            f"    Ensemble OOF R²: {r2_score(y, self.meta_.predict(oof)):.4f}")
        print(f"    Weights — XGB:{self.meta_.coef_[0]:.3f}  "
              f"LGB:{self.meta_.coef_[1]:.3f}  CAT:{self.meta_.coef_[2]:.3f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _safe_X(X)
        return self.meta_.predict(
            np.column_stack([m.predict(X) for m in self.base_])
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. QUANTILE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_quantile_models(X_tr, y_tr, X_te, y_te, label) -> tuple:
    print(f"  Quantile models ({label})...")
    models = {}
    for q, name in [(0.10, "q10"), (0.90, "q90")]:
        m = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            n_estimators=600, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        pred = np.expm1(m.predict(X_te))
        actual = np.expm1(y_te)
        coverage = (pred <= actual).mean() if q == 0.90 else (
            pred >= actual).mean()
        print(f"    {name}: coverage={coverage:.1%}")
        models[name] = m
    return models["q10"], models["q90"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_te, y_te, label: str) -> dict:
    y_pred_log = model.predict(X_te)
    y_pred_real = np.expm1(y_pred_log)
    y_real = np.expm1(y_te)
    r2 = r2_score(y_te, y_pred_log)
    mae = mean_absolute_error(y_real, y_pred_real)
    mape = np.median(np.abs(y_pred_real - y_real) / y_real) * 100

    print(f"\n  ── {label} ──────────────────────────────")
    print(f"  R² (log-space) : {r2:.4f}")
    print(f"  MAE            : ${mae:,.0f} COP")
    print(f"  Median APE     : {mape:.1f}%")

    is_arr = "arr" in label.lower()
    bins = [0, 2e6, 4e6, 6e6, 8e6, 15e6, 1e12] if is_arr else [
        0, 200e6, 400e6, 600e6, 1e9, 2e9, 1e12]
    lbls = ["<2M", "2-4M", "4-6M", "6-8M", "8-15M", ">15M"] if is_arr else \
           ["<200M", "200-400M", "400-600M", "600M-1B", "1-2B", ">2B"]
    bd = pd.DataFrame({"a": y_real, "p": y_pred_real})
    bd["bucket"] = pd.cut(bd["a"], bins=bins, labels=lbls)
    print(bd.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({"n": len(g),
                             "med_APE%": np.median(np.abs(g.p-g.a)/g.a*100)})
    ).to_string())
    return {"r2": r2, "mae": mae, "mape": mape}


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save(obj, name: str):
    p = ARTIFACTS_DIR / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(skip_scrape: bool = False, mode: str = "both", fast: bool = False):
    n_trials = 20 if fast else 50
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"train_{mode}"):
        geo = gpd.read_file("med.shp")

        # Scrape ──────────────────────────────────────────────────────────────
        if skip_scrape:
            print("Loading existing CSVs...")
            arr_raw = pd.read_csv("arriendo_medellin.csv")
            ven_raw = pd.read_csv("venta_medellin.csv")
        else:
            print("\n[Scraping]")
            arr_raw = scrape_all("arriendo")
            ven_raw = scrape_all("venta")
            arr_raw.to_csv("arriendo_medellin.csv", index=False)
            ven_raw.to_csv("venta_medellin.csv",    index=False)

        # Clean ───────────────────────────────────────────────────────────────
        print("\n[1/5] Cleaning...")
        arr_clean = clean(arr_raw, geo, "arriendo")
        ven_clean = clean(ven_raw, geo, "venta")

        # Feature engineering ─────────────────────────────────────────────────
        print("\n[2/5] Engineering features...")
        arr_clean = add_metro_distance(arr_clean)
        ven_clean = add_metro_distance(ven_clean)

        for df in [arr_clean, ven_clean]:
            df["espacios"] = df["habitaciones"] + \
                df["parqueaderos"] + df["baños"]

        ppmc_arr, pppz_arr, pppp_arr = price_aggregates(arr_clean)
        ppmc_ven, pppz_ven, pppp_ven = price_aggregates(ven_clean)

        arr = engineer_features(arr_clean, ppmc_arr, pppz_arr, pppp_arr)
        ven = engineer_features(ven_clean, ppmc_ven, pppz_ven, pppp_ven)

        arr.to_csv("arr_mede_final.csv", index=False)
        ven.to_csv("ven_mede_final.csv", index=False)
        print(f"  arr: {len(arr)} rows | ven: {len(ven)} rows")

        mlflow.log_params({
            "arr_rows": len(arr), "ven_rows": len(ven),
            "n_barrios_arr": arr["nombre"].nunique(),
            "n_barrios_ven": ven["nombre"].nunique(),
            "n_trials": n_trials,
        })

        list_barrios = sorted(arr["nombre"].unique().tolist())

        # Preprocess ──────────────────────────────────────────────────────────
        print("\n[3/5] Preprocessing & feature selection...")

        def prep(df):
            y_full = np.log1p(df["precio"])
            idx_tr, idx_te = train_test_split(
                df.index, test_size=0.2, random_state=1954)
            df_tr, df_te = df.loc[idx_tr].copy(), df.loc[idx_te].copy()
            y_tr,  y_te = y_full.loc[idx_tr], y_full.loc[idx_te]

            # Smoothed target encoding — computed on train split only
            k = 10
            global_mean = y_tr.mean()
            stats = y_tr.groupby(df_tr["nombre"]).agg(["mean", "count"])
            smooth = stats["count"] / (stats["count"] + k)
            te_map = smooth * stats["mean"] + (1 - smooth) * global_mean
            df_tr["barrio_te"] = df_tr["nombre"].map(
                te_map).fillna(global_mean)
            df_te["barrio_te"] = df_te["nombre"].map(
                te_map).fillna(global_mean)

            X_tr, X_te = df_tr[NUM_ATTRIBS +
                               CAT_ATTRIBS], df_te[NUM_ATTRIBS + CAT_ATTRIBS]
            pp = make_preprocessor()
            X_tr_t = pd.DataFrame(pp.fit_transform(
                X_tr), columns=pp.get_feature_names_out())
            X_te_t = pd.DataFrame(pp.transform(
                X_te),     columns=pp.get_feature_names_out())
            cols = select_features(X_tr_t, y_tr, max_features=22)
            return pp, X_tr_t[cols], X_te_t[cols], y_tr, y_te, cols, te_map

        pp_arr, X_tr_arr, X_te_arr, y_tr_arr, y_te_arr, cols_arr, te_arr = prep(
            arr)
        pp_ven, X_tr_ven, X_te_ven, y_tr_ven, y_te_ven, cols_ven, te_ven = prep(
            ven)

        # Optuna tuning ───────────────────────────────────────────────────────
        print("\n[4/5] Hyperparameter tuning (Optuna)...")
        best_arr = best_ven = {}
        if mode in ("both", "arriendo"):
            print("  Arriendo models...")
            best_arr = tune(X_tr_arr, y_tr_arr, n_trials)
        if mode in ("both", "venta"):
            print("  Venta models...")
            best_ven = tune(X_tr_ven, y_tr_ven, n_trials)

        # Train ───────────────────────────────────────────────────────────────
        print("\n[5/5] Training ensembles + quantile models...")
        metrics = {}
        stack_arr = stack_ven = q10_arr = q90_arr = q10_ven = q90_ven = None

        if mode in ("both", "arriendo"):
            stack_arr = StackedEnsemble(
                best_arr["xgb"], best_arr["lgb"], best_arr["cat"]
            ).fit(X_tr_arr, y_tr_arr)
            q10_arr, q90_arr = train_quantile_models(
                X_tr_arr, y_tr_arr, X_te_arr, y_te_arr, "Arriendo")
            metrics["arr"] = evaluate(
                stack_arr, X_te_arr, y_te_arr, "Arriendo Ensemble")

        if mode in ("both", "venta"):
            stack_ven = StackedEnsemble(
                best_ven["xgb"], best_ven["lgb"], best_ven["cat"]
            ).fit(X_tr_ven, y_tr_ven)
            q10_ven, q90_ven = train_quantile_models(
                X_tr_ven, y_tr_ven, X_te_ven, y_te_ven, "Venta")
            metrics["ven"] = evaluate(
                stack_ven, X_te_ven, y_te_ven, "Venta Ensemble")

        # MLflow ──────────────────────────────────────────────────────────────
        for split, m in metrics.items():
            mlflow.log_metrics({f"r2_{split}": m["r2"],
                                f"mae_{split}": m["mae"],
                                f"mape_{split}": m["mape"]})

        # Save ────────────────────────────────────────────────────────────────
        print("\nSaving artifacts...")
        save(pp_arr,       "preprocessor_arr")
        save(pp_ven,       "preprocessor_ven")
        save(cols_arr,     "best_features_arr")
        save(cols_ven,     "best_features_ven")
        save(te_arr,       "barrio_te_arr")
        save(te_ven,       "barrio_te_ven")
        save(stack_arr,    "stack_arr")
        save(stack_ven,    "stack_ven")
        save(q10_arr,      "q10_arr")
        save(q90_arr,      "q90_arr")
        save(q10_ven,      "q10_ven")
        save(q90_ven,      "q90_ven")
        save(ppmc_arr,     "price_per_m2_arr")
        save(ppmc_ven,     "price_per_m2_ven")
        save(pppz_arr,     "price_per_space_arr")
        save(pppz_ven,     "price_per_space_ven")
        save(pppp_arr,     "price_per_parking_arr")
        save(pppp_ven,     "price_per_parking_ven")
        save(list_barrios, "list_barrios")
        save(METRO_STATIONS, "metro_stations")
        save({"arr": metrics.get("arr", {}).get("r2", 0),
              "ven": metrics.get("ven", {}).get("r2", 0)}, "model_r2")

        mlflow.log_artifacts(str(ARTIFACTS_DIR))
        print("\n✓ Done.  Run: mlflow ui   to inspect results.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-scrape", action="store_true")
    p.add_argument("--mode", choices=["both",
                   "arriendo", "venta"], default="both")
    p.add_argument("--fast", action="store_true",
                   help="20 Optuna trials (dev mode)")
    a = p.parse_args()
    run_pipeline(skip_scrape=a.skip_scrape, mode=a.mode, fast=a.fast)
