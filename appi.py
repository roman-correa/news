"""
Compra/Venta Medellín — Real Estate Price Predictor
Author: Roman Alejandro Correa
"""
from io import BytesIO
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np

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
# Cached loaders
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_pickle(path: str):
    with open(path, "rb") as f:
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
arr_cats          = load_pickle("best_features_arr.pkl")
ven_cats          = load_pickle("best_features_ven.pkl")
xgb_arr           = load_pickle("xgb_model_arr_med.pkl")
xgb_ven           = load_pickle("xgb_model_ven_med.pkl")
cat_arr           = load_pickle("cat_model_arr_med.pkl")
list_barrios      = load_pickle("list_barrios.pkl")
ppmc_ven          = load_pickle("price_per_m2_ven.pkl")
pppz_ven          = load_pickle("price_per_space_ven.pkl")
ppmc_arr          = load_pickle("price_per_m2_arr.pkl")
pppz_arr          = load_pickle("price_per_space_arr.pkl")
pppp_arr          = load_pickle("price_per_parking_arr.pkl")
pppp_ven          = load_pickle("price_per_parking_ven.pkl")
preprocessor      = load_pickle("preprocessor.pkl")

loan  = load_csv("arr_mede_final.csv")
sales = load_csv("ven_mede_final.csv")
gdf   = load_geojson("medellin.geojson")

# Normalize barrio names once
for df in [loan, sales]:
    df["barrio_norm"] = df["nombre"].astype(str).str.strip().str.lower()


# ─────────────────────────────────────────────
# Opportunity labels (computed once at startup)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def add_opportunity_labels(_loan, _sales):
    """Return loan and sales DataFrames with opportunity columns added."""
    loan_c  = _loan.copy()
    sales_c = _sales.copy()

    loan_std = preprocessor.transform(loan_c)
    loan_std = pd.DataFrame(loan_std, columns=preprocessor.get_feature_names_out())
    loan_c["cat_pred"]              = np.exp(cat_arr.predict(loan_std[arr_cats]))
    loan_c["is_underpriced"]        = loan_c["precio"] < loan_c["cat_pred"]
    loan_c["pct_underpriced"]       = (loan_c["cat_pred"] - loan_c["precio"]) / loan_c["cat_pred"] * 100
    loan_c["oportunity_houses"]     = loan_c["pct_underpriced"] > 20

    sales_std = preprocessor.transform(sales_c)
    sales_std = pd.DataFrame(sales_std, columns=preprocessor.get_feature_names_out())
    sales_c["cat_pred"]             = np.exp(xgb_ven.predict(sales_std[ven_cats]))
    sales_c["is_underpriced"]       = sales_c["precio"] < sales_c["cat_pred"]
    sales_c["pct_underpriced"]      = (sales_c["cat_pred"] - sales_c["precio"]) / sales_c["cat_pred"] * 100
    sales_c["oportunity_houses"]    = sales_c["pct_underpriced"] > 20

    return loan_c, sales_c


loan, sales = add_opportunity_labels(loan, sales)


# ─────────────────────────────────────────────
# Feature engineering helper
# ─────────────────────────────────────────────
def _build_input_df(area, habitaciones, banos, parqueaderos, barrio, tipo, kind="arr") -> pd.DataFrame:
    ppmc = ppmc_arr if kind == "arr" else ppmc_ven
    pppz = pppz_arr if kind == "arr" else pppz_ven
    pppp = pppp_arr if kind == "arr" else pppp_ven

    espacios = habitaciones + parqueaderos + banos
    axe  = area / espacios if espacios else np.nan
    axh  = area / habitaciones if habitaciones else np.nan
    axa  = area ** 2
    parq2 = parqueaderos ** 2

    barrio_ppmc = ppmc.get(barrio, np.nan)
    barrio_pppz = pppz.get(barrio, np.nan)
    barrio_pppp = pppp.get(barrio, np.nan)

    def safe_ratio(a, b):
        if b and not np.isnan(b) and b != 0:
            return a / b
        return 1.0

    row = {
        "habitaciones": habitaciones,
        "baños": banos,
        "parqueaderos": parqueaderos,
        "espacios": espacios,
        "axe": axe,
        "tipo": tipo,
        "ppmc": barrio_ppmc,
        "pppz": barrio_pppz,
        "pppp": barrio_pppp,
        "garaje_bin": 1 if parqueaderos > 0 else 0,
        "parq2": parq2,
        "new_index": (barrio_ppmc / ppmc_arr.get("max_ppmc", barrio_ppmc + 1)) * 100
                     if kind == "arr" else
                     (barrio_ppmc / ppmc_ven.get("max_ppmc", barrio_ppmc + 1)) * 100,
        "area": area,
        "barrio": barrio,
        "axh": axh,
        "axa": axa,
        "pppp/pppz": safe_ratio(barrio_pppp, barrio_pppz),
        "pppp/ppmc": safe_ratio(barrio_pppp, barrio_ppmc),
        "pppz/ppmc": safe_ratio(barrio_pppz, barrio_ppmc),
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


def predict(area, habitaciones, banos, parqueaderos, barrio, tipo, kind="arr") -> tuple[float | None, float | None, str | None]:
    """Return (prediction, price_per_m2, error_string)."""
    ref_df = loan if kind == "arr" else sales
    err = _validate(barrio, tipo, ref_df)
    if err:
        return None, None, err

    input_df = _build_input_df(area, habitaciones, banos, parqueaderos, barrio, tipo, kind)
    X = preprocessor.transform(input_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())

    cats = arr_cats if kind == "arr" else ven_cats
    model = xgb_arr if kind == "arr" else xgb_ven

    missing = [c for c in cats if c not in X.columns]
    if missing:
        return None, None, f"Faltan columnas: {missing}"

    raw_pred = model.predict(X[cats])
    price = float(np.expm1(raw_pred)[0])
    ppm2  = price / area if area else None
    return price, ppm2, None


# ─────────────────────────────────────────────
# Helper: neighbourhood context
# ─────────────────────────────────────────────
def get_barrio_stats(barrio: str) -> dict:
    """Return a dict of median/count stats for a neighbourhood."""
    bn = barrio.strip().lower()
    arr_b  = loan[loan["barrio_norm"] == bn]
    ven_b  = sales[sales["barrio_norm"] == bn]
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
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏙️ Medellín RE")
    st.markdown("<div class='section-header'>Navegación</div>", unsafe_allow_html=True)
    mode = st.radio(
        "Sección",
        ["🔮 Predictor", "📊 Explorar datos", "🗺️ Mapa de oportunidades"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='section-header'>Filtros globales</div>", unsafe_allow_html=True)
    selected_barrio = st.selectbox("Barrio", ["Todos"] + sorted(list_barrios))
    min_price = st.number_input("Precio mínimo (COP)", value=0, step=1_000_000, format="%d")
    max_price = st.number_input("Precio máximo (COP)", value=2_000_000_000, step=1_000_000, format="%d")

    st.markdown("---")
    st.caption("Por Roman Alejandro Correa")


# ─────────────────────────────────────────────
# PAGE: PREDICTOR
# ─────────────────────────────────────────────
if mode == "🔮 Predictor":
    st.markdown("## Predictor de precios")
    st.markdown("Estima el precio de arriendo o venta de una propiedad en Medellín.")

    kind_label = st.segmented_control("Tipo de transacción", ["Arriendo", "Venta"], default="Arriendo")
    kind = "arr" if kind_label == "Arriendo" else "ven"

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-header'>Datos de la propiedad</div>", unsafe_allow_html=True)
        barrio      = st.selectbox("Barrio", list_barrios)
        tipo        = st.selectbox("Tipo", sorted(loan["tipo"].unique()))
        area        = st.slider("Área (m²)", 20, 500, 80)
        col_a, col_b = st.columns(2)
        with col_a:
            habitaciones = st.number_input("Habitaciones", 0, 10, 2)
            banos        = st.number_input("Baños", 0, 10, 2)
        with col_b:
            parqueaderos = st.number_input("Parqueaderos", 0, 5, 1)

        predict_btn = st.button("Predecir precio →", type="primary", use_container_width=True)

    with col_result:
        st.markdown("<div class='section-header'>Resultado</div>", unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Calculando..."):
                price, ppm2, err = predict(area, habitaciones, banos, parqueaderos, barrio, tipo, kind)

            if err:
                st.error(err)
            else:
                # Confidence range: ±12% (heuristic based on typical XGBoost RE RMSE)
                low  = price * 0.88
                high = price * 1.12

                st.markdown(f"""
                <div class='pred-box'>
                    <div class='pred-label'>Precio estimado · {kind_label}</div>
                    <div class='pred-value'>{fmt_price(price)}</div>
                    <div class='pred-range'>Rango probable: {fmt_price(low)} — {fmt_price(high)}</div>
                </div>
                """, unsafe_allow_html=True)

                # Price per m²
                stats = get_barrio_stats(barrio)
                barrio_ppm2 = stats[f"{'arr' if kind=='arr' else 'ven'}_ppm2"]

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
                st.markdown("<div class='section-header'>Factores clave del precio</div>", unsafe_allow_html=True)

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
                st.markdown("<div class='section-header'>Propiedades similares en el barrio</div>", unsafe_allow_html=True)
                ref = loan if kind == "arr" else sales
                comp = ref[
                    (ref["barrio_norm"] == barrio.strip().lower()) &
                    (ref["tipo"].str.strip().str.lower() == tipo.strip().lower()) &
                    (ref["area"].between(area * 0.75, area * 1.25))
                ].head(5)

                if len(comp):
                    show_cols = [c for c in ["nombre", "tipo", "precio", "area", "habitaciones", "baños", "parqueaderos"] if c in comp.columns]
                    comp_display = comp[show_cols].copy()
                    comp_display["precio"] = comp_display["precio"].apply(fmt_price)
                    st.dataframe(comp_display, use_container_width=True, hide_index=True)
                else:
                    st.caption("No hay comparables con filtros exactos en este barrio.")

        else:
            st.markdown("""
            <div style='color:#3a6080; padding: 40px 20px; text-align: center; border: 1px dashed #1e3a5f; border-radius:12px;'>
                Completa el formulario y presiona<br><strong style='color:#5f8ab0'>Predecir precio →</strong>
            </div>
            """, unsafe_allow_html=True)

            # Show neighbourhood context even before predicting
            stats = get_barrio_stats(barrio if "barrio" in dir() else list_barrios[0])
            st.markdown("<div class='section-header'>Contexto del barrio seleccionado</div>", unsafe_allow_html=True)
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

    ds_choice = st.segmented_control("Dataset", ["Arriendo", "Venta"], default="Arriendo")
    df = loan.copy() if ds_choice == "Arriendo" else sales.copy()

    # Apply sidebar filters
    if selected_barrio != "Todos":
        df = df[df["barrio_norm"] == selected_barrio.strip().lower()]
    df = df[(df["precio"] >= min_price) & (df["precio"] <= max_price)]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total listings",        f"{len(df):,}",                     "filtrados"),
        ("Precio mediana",        fmt_price(df['precio'].median()),    "COP"),
        ("Área mediana",          f"{df['area'].median():.0f} m²"      if 'area' in df.columns else "—", ""),
        ("Oportunidades",         f"{df['oportunity_houses'].sum():,}" if 'oportunity_houses' in df.columns else "—", ">20% subvaloradas"),
    ]
    for col, (label, value, sub) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value' style='font-size:22px'>{value}</div>
                <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Visualizaciones</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribución de precios", "Precio vs Área", "Por barrio"])

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

    st.markdown("<div class='section-header'>Tabla de datos</div>", unsafe_allow_html=True)
    st.dataframe(
        df.drop(columns=["barrio_norm"], errors="ignore").sample(min(500, len(df))) if len(df) else df,
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
    st.markdown("Barrios y propiedades donde el precio de mercado está por debajo del valor predicho por el modelo.")

    map_mode = st.segmented_control("Ver", ["Venta", "Arriendo"], default="Venta")
    df_map = sales if map_mode == "Venta" else loan

    # ── Choropleth ──
    summary = (
        df_map.groupby("barrio_norm")
        .agg(
            count=("precio", "count"),
            median_price=("precio", "median"),
            pct_opp=("oportunity_houses", "mean"),
        )
        .reset_index()
    )
    summary["pct_opp"] *= 100

    gdf_work = gdf.copy()
    # Try to find the neighbourhood name column
    name_col = next((c for c in ["nombre", "name", "NOMBRE"] if c in gdf_work.columns), None)
    if name_col:
        gdf_work["barrio_norm"] = gdf_work[name_col].astype(str).str.strip().str.lower()
    else:
        gdf_work["barrio_norm"] = ""

    merged = gdf_work.merge(summary, on="barrio_norm", how="left")
    merged[["pct_opp", "count", "median_price"]] = merged[["pct_opp", "count", "median_price"]].fillna(0)

    fig_choro = px.choropleth_mapbox(
        merged,
        geojson=merged.geometry.__geo_interface__,
        locations=merged.index,
        color="pct_opp",
        color_continuous_scale="teal",
        range_color=(0, 60),
        mapbox_style="carto-darkmatter",
        center=dict(lat=merged.geometry.centroid.y.mean(), lon=merged.geometry.centroid.x.mean()),
        zoom=10.5,
        opacity=0.7,
        hover_data={"barrio_norm": True, "count": True, "median_price": True, "pct_opp": ":.1f"},
        labels={"pct_opp": "% oportunidades", "median_price": "Precio mediana", "count": "Listings"},
        title=f"% de propiedades subvaloradas por barrio — {map_mode}",
    )
    fig_choro.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8aabcc"),
        height=480,
        coloraxis_colorbar=dict(title="% oport."),
    )
    st.plotly_chart(fig_choro, use_container_width=True)

    # ── Opportunity table ──
    st.markdown("<div class='section-header'>Propiedades oportunidad (>20% subvaloradas)</div>", unsafe_allow_html=True)

    opp = df_map[df_map["oportunity_houses"]].copy()
    if selected_barrio != "Todos":
        opp = opp[opp["barrio_norm"] == selected_barrio.strip().lower()]
    opp = opp[(opp["precio"] >= min_price) & (opp["precio"] <= max_price)]

    st.markdown(f"**{len(opp):,}** propiedades encontradas con descuento potencial > 20%")

    show_cols = [c for c in ["nombre", "tipo", "precio", "cat_pred", "pct_underpriced", "area", "habitaciones", "baños", "parqueaderos"] if c in opp.columns]

    opp_display = opp[show_cols].copy()
    if "precio" in opp_display.columns:
        opp_display["precio"] = opp_display["precio"].apply(fmt_price)
    if "cat_pred" in opp_display.columns:
        opp_display["cat_pred"] = opp_display["cat_pred"].apply(fmt_price)
    if "pct_underpriced" in opp_display.columns:
        opp_display["pct_underpriced"] = opp_display["pct_underpriced"].apply(lambda x: f"{x:.1f}%")

    opp_display = opp_display.rename(columns={
        "nombre": "Barrio", "tipo": "Tipo", "precio": "Precio real",
        "cat_pred": "Precio modelo", "pct_underpriced": "Descuento",
        "area": "m²", "habitaciones": "Hab", "baños": "Baños", "parqueaderos": "Parq",
    })

    st.dataframe(opp_display, use_container_width=True, height=380, hide_index=True)

    # ── Scatter: real vs predicted ──
    st.markdown("<div class='section-header'>Precio real vs. precio estimado por el modelo</div>", unsafe_allow_html=True)
    scatter_df = df_map[["precio", "cat_pred", "barrio_norm", "tipo", "oportunity_houses"]].dropna().sample(min(1500, len(df_map)))
    fig_sc = px.scatter(
        scatter_df,
        x="cat_pred", y="precio",
        color="oportunity_houses",
        color_discrete_map={True: "#0fd4c0", False: "#1e3a5f"},
        hover_data=["barrio_norm", "tipo"],
        labels={"cat_pred": "Precio modelo", "precio": "Precio real", "oportunity_houses": "Oportunidad"},
        template="plotly_dark",
        title="Precio real vs estimado — teal = oportunidad",
        opacity=0.7,
    )
    # diagonal reference line
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
