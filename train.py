"""
Medellín Real Estate — Training Pipeline
=========================================
Scrapes metrocuadrado.com, cleans data, engineers features,
trains XGBoost models for arriendo and venta, and saves all
artifacts needed by app.py.

Usage:
    python train.py                        # scrape + train both
    python train.py --skip-scrape          # use existing CSVs
    python train.py --mode arriendo        # train only arriendo
    python train.py --mode venta           # train only venta

Outputs (all saved to ./artifacts/):
    preprocessor_arr.pkl, preprocessor_ven.pkl
    xgb_model_arr.pkl,    xgb_model_ven.pkl
    best_features_arr.pkl, best_features_ven.pkl
    price_per_m2_arr.pkl,  price_per_m2_ven.pkl
    price_per_space_arr.pkl, price_per_space_ven.pkl
    price_per_parking_arr.pkl, price_per_parking_ven.pkl
    list_barrios.pkl
    arr_mede_final.csv,   ven_mede_final.csv
"""

import argparse
import os
import pickle
import time
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Medellín bounding box (lat/lon) — filters out bad geocodes
MEDELLIN_BOUNDS = dict(lat_min=6.175, lat_max=6.28,
                       lon_min=-75.62, lon_max=-75.55)

NUM_ATTRIBS = [
    # Property size
    "area", "habitaciones", "baños", "parqueaderos", "espacios",
    # Area interactions
    "axe", "axh", "axa", "parq2", "garaje_bin",
    # Neighbourhood price signals (aggregate market rates — no leakage)
    "ppmc", "pppz", "pppp", "new_index",
    # Neighbourhood ratios
    "pppp/pppz", "pppp/ppmc", "pppz/ppmc",
    # Listing density (count only — not price-derived)
    "barrio_count",
    # Target encoding (smoothed, train-only — no leakage)
    "barrio_te",
]
CAT_ATTRIBS = ["tipo"]

# ─────────────────────────────────────────────
# 1. SCRAPER
# ─────────────────────────────────────────────


def get_headers() -> dict:
    """Minimal headers needed for metrocuadrado API."""
    return {
        "accept": "*/*",
        "content-type": "application/json",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "x-api-key": "P1MfFHfQMOtL16Zpg36NcntJYCLFm8FqFfudnavl",
    }


def scrape(operacion: str = "arriendo", ciudad: str = "medellin",
           num_paginas: int = 201, delay: float = 0.3) -> pd.DataFrame:
    """
    Scrape metrocuadrado.com listings.

    Parameters
    ----------
    operacion   : 'arriendo' or 'venta'
    ciudad      : city slug
    num_paginas : pages of 50 results each (201 → ~10 000 listings)
    delay       : seconds between requests (be polite)
    """
    headers = get_headers()
    records = []

    print(f"Scraping {operacion} — {num_paginas} pages × 50 results...")
    for page in range(num_paginas):
        params = {
            "size": "50",
            "from": str(page * 50),
            "realEstateBusinessList": operacion,
            "city": ciudad,
        }
        try:
            r = requests.get(
                "https://www.metrocuadrado.com/rest-search/search",
                params=params,
                headers=headers,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  Page {page}: error — {e}. Skipping.")
            continue

        results = data.get("results", [])
        if not results:
            print(f"  Page {page}: no results, stopping early.")
            break

        for item in results:
            def g(key, default=None):
                try:
                    return item.get(key)
                except Exception:
                    return default

            precio = (
                g("mvalorarriendo") if operacion == "arriendo" else g("mvalorventa")
            )
            loc = g("localizacion") or {}
            tipo_obj = g("mtipoinmueble") or {}
            link = g("link", "")

            records.append({
                "barrio":        g("mnombrecomunbarrio"),
                "barrio_1":      g("mbarrio"),
                "precio":        precio,
                "area":          g("marea"),
                "habitaciones":  g("mnrocuartos"),
                "baños":         g("mnrobanos"),
                "parqueaderos":  g("mnrogarajes"),
                "tipo":          tipo_obj.get("nombre"),
                "latitud":       loc.get("lat"),
                "longitud":      loc.get("lon"),
                "url":           "https://www.metrocuadrado.com" + link if link else None,
            })

        if (page + 1) % 20 == 0:
            print(
                f"  {page + 1}/{num_paginas} pages done, {len(records)} records so far")
        time.sleep(delay)

    df = pd.DataFrame(records)
    # Keep only Apartamento and Casa
    df = df[df["tipo"].str.contains("Apartamento|Casa", na=False, case=False)]
    print(f"Scraped {len(df)} {operacion} listings.")
    return df


# ─────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────

def normalize_text(s: str) -> str:
    """Lowercase + strip accents — used ONLY for join keys, never for display values."""
    s = str(s).strip().lower()
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ASCII", "ignore")
        .decode("ASCII")
    )


def clean_nombre(s: str) -> str:
    """Keep the original shapefile casing — strip whitespace and expansion prefix only."""
    s = str(s).strip()
    s = s.replace("Área de Expansión", "").replace(
        "Area de Expansion", "").strip()
    # Remove leading/trailing punctuation left after prefix removal
    return s.strip(" -–—")


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose coordinates are missing or fall outside Medellín bounds.

    We no longer try to rescue bad coordinates by guessing from the scraped
    barrio string — that string is unreliable and causes the spatial join to
    assign the wrong polygon name. Real coordinates only; everything else drops.
    """
    b = MEDELLIN_BOUNDS
    before = len(df)
    df = df.dropna(subset=["latitud", "longitud"])
    df = df[
        (df["latitud"] >= b["lat_min"]) & (df["latitud"] <= b["lat_max"]) &
        (df["longitud"] >= b["lon_min"]) & (df["longitud"] <= b["lon_max"])
    ].copy()
    dropped = before - len(df)
    if dropped:
        print(
            f"    fix_coordinates: dropped {dropped} rows with bad/missing coordinates")
    return df.reset_index(drop=True)


def match_barrio_names(df: pd.DataFrame, geo: gpd.GeoDataFrame,
                       threshold: int = 80) -> pd.DataFrame:
    """Fuzzy-match scraped barrio names to official shapefile names."""
    ref_names = geo["NOMBRE"].astype(
        str).str.strip().str.lower().unique().tolist()
    df["barrio"] = df["barrio"].astype(str).str.strip().str.lower()

    def best_match(name):
        result = process.extractOne(
            name, ref_names, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            return result[0]
        return name

    df["barrio"] = df["barrio"].apply(best_match)
    return df


def clean(df: pd.DataFrame, geo: gpd.GeoDataFrame, operacion: str) -> pd.DataFrame:
    """Full cleaning pipeline for one dataset."""
    print(f"  Cleaning {operacion}: {len(df)} rows...")

    # Drop rows missing critical fields
    df = df.dropna(subset=["precio", "area", "habitaciones", "baños"])
    df = df[df["precio"] > 0]
    df = df[df["area"] > 0]

    # Cast types
    for col in ["habitaciones", "baños", "parqueaderos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            0).apply(np.floor).astype(int)

    # Keep only Apartamento / Casa
    df["tipo"] = df["tipo"].str.strip().str.lower()
    df = df[df["tipo"].str.contains("apartamento|casa", na=False)]

    # Fix coordinates
    df = fix_coordinates(df)

    # Spatial join to get official barrio name
    gdf_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]), crs="EPSG:4326"
    )
    geo_join = gdf_points.sjoin(geo.to_crs("EPSG:4326"), how="left")
    # sjoin can duplicate rows for points on polygon boundaries — keep first match only
    geo_join = geo_join[~geo_join.index.duplicated(keep="first")]
    geo_join = geo_join.reset_index(drop=True)
    geo_join.columns = geo_join.columns.str.lower()
    df = pd.DataFrame(geo_join).reset_index(drop=True)

    # Ensure unique index before any column assignment
    df = df.reset_index(drop=True)

    # Use ONLY the shapefile NOMBRE — the authoritative neighbourhood name.
    # geo_join.columns.str.lower() made it "nombre" but the VALUE is still the
    # original mixed-case string from the shapefile (e.g. "El Poblado").
    # We keep the original casing — only strip whitespace and expansion prefix.
    df["nombre"] = df["nombre"].astype(str).apply(clean_nombre)

    # Drop rows the sjoin couldn't assign to any polygon
    df = df[df["nombre"].str.len() > 0]
    df = df[~df["nombre"].str.lower().isin(["nan", "none", ""])]

    # ── Outlier removal ─────────────────────────────────────────────────────
    # Use a wider IQR fence (3× instead of 1.5×) so high-end properties
    # are NOT removed — they're real data, not errors.
    # Also enforce a sensible absolute minimum.
    if operacion == "arriendo":
        # floor: no real rental below 500k COP
        df = df[df["precio"] >= 500_000]
    else:
        # floor: no real sale below 50M COP
        df = df[df["precio"] >= 50_000_000]

    Q1 = df["precio"].quantile(0.25)
    Q3 = df["precio"].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3.0 * IQR   # wide fence — keep luxury properties
    df = df[df["precio"] <= upper]

    # Area outliers (data entry errors: 5000 m² apartments)
    df = df[df["area"] <= df["area"].quantile(0.99)]

    df = df[["tipo", "precio", "area", "habitaciones", "baños",
             "parqueaderos", "nombre", "latitud", "longitud", "url"]].dropna(subset=["nombre"])

    print(f"  After cleaning: {len(df)} rows | precio range: "
          f"{df['precio'].min():,.0f} → {df['precio'].max():,.0f}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def price_aggregates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute neighbourhood-level price aggregates."""
    precio_tot = df.groupby("nombre")["precio"].sum().astype(float)
    area_tot = df.groupby("nombre")["area"].sum().replace(
        0, np.nan).astype(float)
    espacios = df.groupby("nombre")["espacios"].sum().astype(float)
    parq = (df.groupby("nombre")["parqueaderos"].sum() + 1).astype(float)

    ppmc = (precio_tot / area_tot).dropna().sort_values(ascending=False)
    pppz = (precio_tot / espacios).dropna().sort_values(ascending=False)
    pppp = (precio_tot / parq).dropna().sort_values(ascending=False)
    return ppmc, pppz, pppp


def engineer_features(df: pd.DataFrame,
                      ppmc: pd.Series,
                      pppz: pd.Series,
                      pppp: pd.Series) -> pd.DataFrame:
    """Add all engineered features. barrio_te is NOT added here to avoid leakage
    — it is computed train-only inside prep_dataset and mapped to test rows."""
    df = df.copy()

    df["espacios"] = df["habitaciones"] + df["parqueaderos"] + df["baños"]

    # Per-unit area ratios
    df["axe"] = np.where(df["espacios"] == 0, df["area"],
                         df["area"] / df["espacios"])
    df["axh"] = np.where(df["habitaciones"] == 0, df["area"],
                         df["area"] / df["habitaciones"])
    df["axa"] = df["area"] ** 2
    df["parq2"] = df["parqueaderos"] ** 2
    df["garaje_bin"] = (df["parqueaderos"] > 0).astype(int)

    # Neighbourhood price signals
    df["ppmc"] = df["nombre"].map(ppmc)
    df["pppz"] = df["nombre"].map(pppz)
    df["pppp"] = df["nombre"].map(pppp)

    ppmc_max = ppmc.max()
    df["new_index"] = df["ppmc"] / ppmc_max * 100

    def safe_ratio(a, b):
        return (a / b).replace([np.inf, -np.inf], np.nan).fillna(1).astype(float)

    df["pppp/pppz"] = safe_ratio(df["pppp"], df["pppz"])
    df["pppp/ppmc"] = safe_ratio(df["pppp"], df["ppmc"])
    df["pppz/ppmc"] = safe_ratio(df["pppz"], df["ppmc"])

    # Listing density — count of properties per barrio (no leakage)
    df["barrio_count"] = df["nombre"].map(
        df.groupby("nombre")["precio"].count())

    # Placeholder for barrio_te — filled correctly in prep_dataset (train-only)
    df["barrio_te"] = np.nan

    # Drop rows where neighbourhood aggregates are missing
    df = df.dropna(subset=["ppmc", "pppz", "pppp", "pppz/ppmc"])

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. FEATURE SELECTION
# ─────────────────────────────────────────────

def select_features(X_train: pd.DataFrame, y_train: pd.Series,
                    max_features: int = 10, cv: int = 5) -> list[str]:
    """
    Select best feature subset using XGBoost feature importances
    (much faster and more reliable than exhaustive LinearRegression search).
    """
    from sklearn.model_selection import cross_val_score

    print("  Selecting features via XGBoost importances...")

    # Fit a quick XGBoost on all features
    probe = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, random_state=42)
    probe.fit(X_train, y_train)

    importances = pd.Series(probe.feature_importances_, index=X_train.columns)
    # Take top max_features by importance
    top_features = importances.nlargest(max_features).index.tolist()

    # Validate with cross-val R²
    cv_r2 = cross_val_score(probe, X_train[top_features], y_train,
                            cv=cv, scoring="r2").mean()
    print(f"  Selected {len(top_features)} features | CV R²: {cv_r2:.4f}")
    print(f"  Features: {top_features}")
    return top_features


# ─────────────────────────────────────────────
# 5. PREPROCESSOR
# ─────────────────────────────────────────────

def make_preprocessor() -> ColumnTransformer:
    """Create a fresh (unfitted) preprocessor — call once per dataset."""
    return ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), NUM_ATTRIBS),
        ("cat", Pipeline(
            [("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT_ATTRIBS),
    ])


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 6. TRAINING  (Bayesian hyperparameter search + early stopping refit)
# ─────────────────────────────────────────────

# Search space — 8 dimensions covering the parameters that matter most
# for gradient-boosted trees on ~7k rows
XGB_SEARCH_SPACE = {
    "learning_rate":    Real(0.005, 0.1,  prior="log-uniform"),
    "max_depth":        Integer(3, 6),
    "min_child_weight": Integer(4, 20),
    "subsample":        Real(0.6, 0.95,  prior="uniform"),
    "colsample_bytree": Real(0.5, 0.95,  prior="uniform"),
    "gamma":            Real(0.0, 0.5,   prior="uniform"),
    "reg_alpha":        Real(1e-3, 5.0,  prior="log-uniform"),
    "reg_lambda":       Real(0.5, 8.0,   prior="log-uniform"),
}


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame,  y_test: pd.Series,
              label: str,
              n_iter: int = 40,
              cv: int = 5) -> XGBRegressor:
    """
    Two-stage training:
      1. BayesSearchCV explores the param space with n_iter evaluations
         using n_estimators=500 per candidate (fast).
      2. Best params refitted with n_estimators=2000 + early stopping
         to find the exact tree count.

    n_iter=40 → ~5-10 min on a laptop. Raise to 60-80 for a deeper search.
    """
    print(
        f'  Bayesian search ({label}): {n_iter} iterations x {cv}-fold CV...')

    # Stage 1: search
    base = XGBRegressor(
        n_estimators=500,
        eval_metric="rmse",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    search = BayesSearchCV(
        base,
        XGB_SEARCH_SPACE,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        refit=False,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_params = dict(search.best_params_)

    print(f'  Best CV R2     : {search.best_score_:.4f}')
    print(f'  Best params:')
    for k, v in sorted(best_params.items()):
        print(f'    {k}: {v}')

    # Stage 2: refit with early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    model = XGBRegressor(
        **best_params,
        n_estimators=2000,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    y_pred_log = model.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)
    r2 = r2_score(y_test, y_pred_log)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mape = np.median(np.abs(y_pred_real - y_test_real) / y_test_real) * 100

    print(f'\n  -- {label} final results --')
    print(f'  Best iteration : {model.best_iteration}')
    print(f'  R2 (log space) : {r2:.4f}')
    print(f'  MAE (COP)      : ${mae:,.0f}')
    print(f'  Median APE     : {mape:.1f}%')

    bucket_df = pd.DataFrame({'actual': y_test_real, 'pred': y_pred_real})
    bins = [0, 2e6, 4e6, 6e6, 8e6, 15e6, 1e12] if "arr" in label.lower() else \
        [0, 200e6, 400e6, 600e6, 1e9, 2e9, 1e12]
    labels_b = ["<2M", "2-4M", "4-6M", "6-8M", "8-15M", ">15M"] if "arr" in label.lower() else \
               ["<200M", "200-400M", "400-600M", "600M-1B", "1-2B", ">2B"]
    bucket_df['bucket'] = pd.cut(
        bucket_df['actual'], bins=bins, labels=labels_b)
    breakdown = bucket_df.groupby('bucket', observed=True).apply(
        lambda g: pd.Series({
            'n': len(g),
            'median_APE%': np.median(np.abs(g['pred'] - g['actual']) / g['actual'] * 100)
        })
    )
    print(f'\n  Error by price bucket:\n{breakdown.to_string()}\n')

    return model


# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save(obj, name: str):
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# 8. FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(skip_scrape: bool = False, mode: str = "both"):
    geo = gpd.read_file("med.shp")

    # ── Scrape or load ───────────────────────────────────────────────────────
    if skip_scrape:
        print("Loading existing CSVs...")
        arr_raw = pd.read_csv("arriendo_medellin.csv")
        ven_raw = pd.read_csv("venta_medellin.csv")
    else:
        arr_raw = scrape("arriendo", num_paginas=201)
        ven_raw = scrape("venta",    num_paginas=201)
        arr_raw.to_csv("arriendo_medellin.csv", index=False)
        ven_raw.to_csv("venta_medellin.csv",    index=False)

    # ── Clean ────────────────────────────────────────────────────────────────
    print("\n[1/4] Cleaning...")
    arr_clean = clean(arr_raw, geo, "arriendo")
    ven_clean = clean(ven_raw, geo, "venta")

    # ── Feature engineering ──────────────────────────────────────────────────
    print("\n[2/4] Engineering features...")
    ppmc_arr, pppz_arr, pppp_arr = price_aggregates(
        arr_clean.assign(
            espacios=arr_clean["habitaciones"] + arr_clean["parqueaderos"] + arr_clean["baños"])
    )
    ppmc_ven, pppz_ven, pppp_ven = price_aggregates(
        ven_clean.assign(
            espacios=ven_clean["habitaciones"] + ven_clean["parqueaderos"] + ven_clean["baños"])
    )

    arr = engineer_features(arr_clean, ppmc_arr, pppz_arr, pppp_arr)
    ven = engineer_features(ven_clean, ppmc_ven, pppz_ven, pppp_ven)

    # Save processed data
    arr.to_csv("arr_mede_final.csv", index=False)
    ven.to_csv("ven_mede_final.csv", index=False)
    print(
        f"  arr_mede_final: {len(arr)} rows | ven_mede_final: {len(ven)} rows")

    list_barrios = sorted(arr["nombre"].unique().tolist())

    # ── Preprocess + select features ─────────────────────────────────────────
    print("\n[3/4] Preprocessing & feature selection...")

    def prep_dataset(df, label):
        y_full = np.log1p(df["precio"])

        # Split BEFORE computing barrio_te to prevent leakage
        idx_tr, idx_te = train_test_split(
            df.index, test_size=0.2, random_state=1954
        )
        df_tr = df.loc[idx_tr].copy()
        df_te = df.loc[idx_te].copy()
        y_tr = y_full.loc[idx_tr]
        y_te = y_full.loc[idx_te]

        # Smoothed target encoding — computed on train rows only.
        # Blends barrio mean with global mean: smoothing = count / (count + k)
        # k=10 means barrios with <10 listings lean heavily on the global mean.
        k = 10
        global_mean = y_tr.mean()
        barrio_stats = y_tr.groupby(df_tr["nombre"]).agg(["mean", "count"])
        smoothing = barrio_stats["count"] / (barrio_stats["count"] + k)
        barrio_te_map = smoothing * \
            barrio_stats["mean"] + (1 - smoothing) * global_mean
        df_tr["barrio_te"] = df_tr["nombre"].map(
            barrio_te_map).fillna(global_mean)
        df_te["barrio_te"] = df_te["nombre"].map(
            barrio_te_map).fillna(global_mean)

        X_tr = df_tr[NUM_ATTRIBS + CAT_ATTRIBS]
        X_te = df_te[NUM_ATTRIBS + CAT_ATTRIBS]

        # SEPARATE preprocessor per dataset
        pp = make_preprocessor()
        X_tr_t = pd.DataFrame(pp.fit_transform(
            X_tr), columns=pp.get_feature_names_out())
        X_te_t = pd.DataFrame(pp.transform(
            X_te),     columns=pp.get_feature_names_out())

        best_cols = select_features(X_tr_t, y_tr, max_features=20)

        return pp, X_tr_t, X_te_t, y_tr, y_te, best_cols, barrio_te_map

    pp_arr, X_tr_arr, X_te_arr, y_tr_arr, y_te_arr, best_cols_arr, barrio_te_arr = prep_dataset(
        arr, "arriendo")
    pp_ven, X_tr_ven, X_te_ven, y_tr_ven, y_te_ven, best_cols_ven, barrio_te_ven = prep_dataset(
        ven, "venta")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[4/4] Training models...")

    if mode in ("both", "arriendo"):
        print("\nTraining arriendo model...")
        model_arr = train_xgb(X_tr_arr[best_cols_arr], y_tr_arr,
                              X_te_arr[best_cols_arr], y_te_arr, "Arriendo")

    if mode in ("both", "venta"):
        print("\nTraining venta model...")
        model_ven = train_xgb(X_tr_ven[best_cols_ven], y_tr_ven,
                              X_te_ven[best_cols_ven], y_te_ven, "Venta")

    # ── Save all artifacts ───────────────────────────────────────────────────
    print("\nSaving artifacts...")
    save(pp_arr,         "preprocessor_arr")
    save(pp_ven,         "preprocessor_ven")
    save(best_cols_arr,  "best_features_arr")
    save(best_cols_ven,  "best_features_ven")
    save(model_arr,      "xgb_model_arr_med")
    save(model_ven,      "xgb_model_ven_med")
    save(ppmc_arr,       "price_per_m2_arr")
    save(ppmc_ven,       "price_per_m2_ven")
    save(pppz_arr,       "price_per_space_arr")
    save(pppz_ven,       "price_per_space_ven")
    save(pppp_arr,       "price_per_parking_arr")
    save(pppp_ven,       "price_per_parking_ven")
    save(list_barrios,   "list_barrios")
    save(barrio_te_arr,  "barrio_te_arr")
    save(barrio_te_ven,  "barrio_te_ven")

    print("\n✓ Pipeline complete. All artifacts saved to ./artifacts/")
    print("  Update app.py to load from ./artifacts/ and use preprocessor_arr / preprocessor_ven.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medellín RE training pipeline")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping, use existing arriendo/venta_medellin.csv")
    parser.add_argument("--mode", choices=["both", "arriendo", "venta"],
                        default="both", help="Which model(s) to train")
    args = parser.parse_args()

    run_pipeline(skip_scrape=args.skip_scrape, mode=args.mode)
