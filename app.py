from io import BytesIO
import streamlit as st
import pandas as pd

import geopandas as gpd
import plotly.express as px
import pickle
import pandas as pd
import numpy as np

# load the pickled object safely from file

loan = pd.read_csv('arr_mede_final.csv')
sales = pd.read_csv('ven_mede_final.csv')
gdf = gpd.read_file("medellin.geojson")

# -----------------------------
# Helpers for caching loads
# -----------------------------

@st.cache_data(ttl=3600)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_geojson(path):
    return gpd.read_file(path)


# -----------------------------
# Load model artifacts and data
# -----------------------------
arr_cats = load_pickle("best_features_arr.pkl")
ven_cats = load_pickle('best_features_ven.pkl')
xgb_model_arr_med = load_pickle('xgb_model_arr_med.pkl')
xgb_model_ven_med = load_pickle('xgb_model_ven_med.pkl')
list_barrios = load_pickle('list_barrios.pkl')
ppmc_ven = load_pickle('price_per_m2_ven.pkl')
pppz_ven = load_pickle('price_per_space_ven.pkl')
ppmc_arr = load_pickle('price_per_m2_arr.pkl')
pppz_arr = load_pickle('price_per_space_arr.pkl')
preprocessor = load_pickle('preprocessor.pkl')
pppp_arr = load_pickle('price_per_parking_arr.pkl')
pppp_ven = load_pickle('price_per_parking_ven.pkl')
cat_model_arr_med = load_pickle('cat_model_arr_med.pkl')

loan = load_csv('arr_mede_final.csv')
sales = load_csv('ven_mede_final.csv')
gdf = load_geojson('medellin.geojson')

# Normalize barrio names in dataframes for safe merges
loan['barrio_norm'] = loan['nombre'].astype(str).str.strip().str.lower()
sales['barrio_norm'] = sales['nombre'].astype(str).str.strip().str.lower()

# -----------------------------
# Prediction functions (robust)
# -----------------------------


def _prepare_input(area, habitaciones, banos, parqueaderos, barrio, tipo, kind='arr'):
    # generic input DataFrame and feature engineering shared by both models
    input_df = pd.DataFrame([{
        'habitaciones': habitaciones,
        'baños': banos,
        'parqueaderos': parqueaderos,
        'espacios': None,
        'axe': None,
        'tipo': tipo,
        'ppmc': None,
        'pppz': None,
        'garaje_bin': None,
        'parq2': None,
        'new_index': None,
        'area': area,
        'barrio': barrio,
        'axh': None,
        'axa': None
    }])

    input_df['espacios'] = input_df['habitaciones'] + \
        input_df['parqueaderos'] + input_df['baños']
    input_df['axe'] = input_df['area'] / \
        input_df['espacios'].replace({0: np.nan})
    input_df['axh'] = input_df['area'] / \
        input_df['habitaciones'].replace({0: np.nan})
    input_df['axa'] = input_df['area'] * input_df['area']
    input_df['parq2'] = input_df['parqueaderos'] * input_df['parqueaderos']

    if kind == 'arr':
        input_df['ppmc'] = input_df['barrio'].map(ppmc_arr)
        input_df['pppz'] = input_df['barrio'].map(pppz_arr)
        input_df['pppp'] = input_df['barrio'].map(pppp_arr)
    else:
        input_df['ppmc'] = input_df['barrio'].map(ppmc_ven)
        input_df['pppz'] = input_df['barrio'].map(pppz_ven)
        input_df['pppp'] = input_df['barrio'].map(pppp_ven)

    input_df['new_index'] = input_df['ppmc']/(input_df['ppmc'].max())*100
    input_df['garaje_bin'] = input_df['parqueaderos'].apply(
        lambda x: 1 if x > 0 else 0)

    input_df['pppp/pppz'] = (input_df['pppp'] / input_df['pppz']
                             ).replace([np.inf, -np.inf], 0).fillna(1).astype(float)
    input_df['pppp/ppmc'] = (input_df['pppp'] / input_df['ppmc']
                             ).replace([np.inf, -np.inf], 0).fillna(1).astype(float)
    input_df['pppz/ppmc'] = (input_df['pppz'] / input_df['ppmc']
                             ).replace([np.inf, -np.inf], 0).fillna(1).astype(float)

    return input_df


def predict_arr(area, habitaciones, banos, parqueaderos, barrio, tipo):
    barrio_norm = str(barrio).strip().lower()
    barrios_lc = [b.strip().lower() for b in list_barrios]
    if barrio_norm not in barrios_lc:
        return None, f'Barrio "{barrio}" no está en la lista.'

    tipo_norm = str(tipo).strip().lower()
    tipos_lc = [t.strip().lower() for t in loan['tipo'].unique().tolist()]
    if tipo_norm not in tipos_lc:
        return None, f'Tipo "{tipo}" no está en la lista.'

    input_df = _prepare_input(
        area, habitaciones, banos, parqueaderos, barrio, tipo, kind='arr')
    X = preprocessor.transform(input_df)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    feature_names = preprocessor.get_feature_names_out()
    X = pd.DataFrame(X, columns=feature_names)

    missing_cols = [c for c in arr_cats if c not in X.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas: {missing_cols}")

    pred = xgb_model_arr_med.predict(X[arr_cats])
    return float(np.expm1(pred)[0]), None


def predict_ven(area, habitaciones, banos, parqueaderos, barrio, tipo):
    barrio_norm = str(barrio).strip().lower()
    barrios_lc = [b.strip().lower() for b in list_barrios]
    if barrio_norm not in barrios_lc:
        return None, f'Barrio "{barrio}" no está en la lista.'

    tipo_norm = str(tipo).strip().lower()
    tipos_lc = [t.strip().lower() for t in sales['tipo'].unique().tolist()]
    if tipo_norm not in tipos_lc:
        return None, f'Tipo "{tipo}" no está en la lista.'

    input_df = _prepare_input(
        area, habitaciones, banos, parqueaderos, barrio, tipo, kind='ven')
    X = preprocessor.transform(input_df)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    feature_names = preprocessor.get_feature_names_out()
    X = pd.DataFrame(X, columns=feature_names)

    missing_cols = [c for c in ven_cats if c not in X.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas: {missing_cols}")

    pred = xgb_model_ven_med.predict(X[ven_cats])
    return float(np.expm1(pred)[0]), None


# -----------------------------
# Opportunity labels (adds columns if not already present)
# -----------------------------
if 'cat_pred' not in loan.columns:
    loan_std = preprocessor.transform(loan)
    loan_std = pd.DataFrame(
        loan_std, columns=preprocessor.get_feature_names_out())
    loan['cat_pred'] = np.exp(cat_model_arr_med.predict(loan_std[arr_cats]))
    loan['is underpriced_cat'] = loan['precio'] < loan['cat_pred']
    loan['how much underpriced_cat'] = (
        loan['cat_pred'] - loan['precio'])/loan['cat_pred']*100
    loan['oportunity_houses'] = loan['how much underpriced_cat'] > 20

if 'cat_pred' not in sales.columns:
    sales_std = preprocessor.transform(sales)
    sales_std = pd.DataFrame(
        sales_std, columns=preprocessor.get_feature_names_out())
    sales['cat_pred'] = np.exp(xgb_model_ven_med.predict(sales_std[ven_cats]))
    sales['is underpriced_cat'] = sales['precio'] < sales['cat_pred']
    sales['how much underpriced_cat'] = (
        sales['cat_pred'] - sales['precio'])/sales['cat_pred']*100
    sales['oportunity_houses'] = sales['how much underpriced_cat'] > 20

# -----------------------------
# UI: Theme + sidebar
# -----------------------------
st.set_page_config(page_title='Medellín Prices - Dashboard', layout='wide')

# dark mode styling (minimal)
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1724; color: #e6eef8; }
    .stButton>button { background-color: #0ea5a4; }
    .card { background-color: #0b1220; padding: 12px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.title('Controles')
mode = st.sidebar.radio(
    'Modo', ['Arriendo', 'Venta', 'Explorar datos', 'Mapa de oportunidades'])


# Filters common to explorers
selected_barrio = st.sidebar.selectbox(
    'Barrio (filtro)', ['Todos'] + sorted(list_barrios))
min_price = st.sidebar.number_input('Precio mínimo', value=0, step=1000000)
max_price = st.sidebar.number_input(
    'Precio máximo', value=1000000000, step=1000000)

# Quick input for prediction
st.sidebar.markdown('---')
st.sidebar.subheader('Predictor rápido')
quick_mode = st.sidebar.radio('Tipo', ['Arriendo', 'Venta'])
quick_barrio = st.sidebar.selectbox(
    'Barrio (predictor)', list_barrios, key='quick_barrio')
quick_tipo = st.sidebar.selectbox('Tipo (predictor)', sorted(
    loan['tipo'].unique()), key='quick_tipo')
quick_area = st.sidebar.number_input(
    'Área m²', min_value=10, max_value=1000, value=80, key='quick_area')
quick_habs = st.sidebar.number_input(
    'Habitaciones', 0, 10, 2, key='quick_habs')
quick_banos = st.sidebar.number_input('Baños', 0, 10, 2, key='quick_banos')
quick_parq = st.sidebar.number_input('Parqueaderos', 0, 5, 1, key='quick_parq')

if st.sidebar.button('Predecir (rápido)'):
    if quick_mode == 'Arriendo':
        val, err = predict_arr(
            quick_area, quick_habs, quick_banos, quick_parq, quick_barrio, quick_tipo)
    else:
        val, err = predict_ven(
            quick_area, quick_habs, quick_banos, quick_parq, quick_barrio, quick_tipo)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f'Precio estimado: ${val:,.0f}')

# -----------------------------
# Main: different modes
# -----------------------------
if mode in ['Arriendo', 'Venta']:
    st.header(f'Predictor - {mode}')
    col1, col2 = st.columns([1, 2])
    with col1:
        barrio = st.selectbox('Barrio', list_barrios)
        tipo = st.selectbox('Tipo', sorted(loan['tipo'].unique()))
        area = st.number_input('Área (m²)', 20, 1000, 80)
        habitaciones = st.number_input('Habitaciones', 0, 10, 2)
        banos = st.number_input('Baños', 0, 10, 2)
        parqueaderos = st.number_input('Parqueaderos', 0, 5, 1)
        if st.button('🔮 Predecir precio'):
            if mode == 'Arriendo':
                pred, err = predict_arr(
                    area, habitaciones, banos, parqueaderos, barrio, tipo)
            else:
                pred, err = predict_ven(
                    area, habitaciones, banos, parqueaderos, barrio, tipo)
            if err:
                st.error(err)
            else:
                st.success(f'Precio estimado: **${pred:,.0f}**')
                st.info(f'Precio por m²: ${pred/area:,.0f} /m²')
    with col2:
        st.subheader('Contexto del barrio')
        # neighborhood statistics summary
        barrio_norm = barrio.strip().lower()
        arr_b = loan[loan['nombre'].astype(
            str).str.strip().str.lower() == barrio_norm]
        ven_b = sales[sales['nombre'].astype(
            str).str.strip().str.lower() == barrio_norm]
        st.metric('Listings arriendo (count)', len(arr_b))
        st.metric('Listings venta (count)', len(ven_b))
        if len(arr_b):
            st.write('Arriendo - precio mediana:',
                     f"${arr_b['precio'].median():,.0f}")
        if len(ven_b):
            st.write('Venta - precio mediana:',
                     f"${ven_b['precio'].median():,.0f}")

elif mode == 'Explorar datos':
    st.header('Explorador de datos')
    ds_choice = st.selectbox('Dataset', ['Arriendo', 'Venta'])
    df = loan.copy() if ds_choice == 'Arriendo' else sales.copy()

    # Apply sidebar filters
    if selected_barrio != 'Todos':
        df = df[df['nombre'].str.strip().str.lower() ==
                selected_barrio.strip().lower()]
    df = df[(df['precio'] >= min_price) & (df['precio'] <= max_price)]

    st.subheader(f'{ds_choice} - {len(df)} rows')
    st.dataframe(df.sample(min(500, len(df))) if len(df) > 0 else df)

    st.markdown('### Visualizaciones')
    c1, c2 = st.columns(2)
    with c1:
        if len(df):
            fig = px.histogram(df, x='precio', nbins=40,
                               title='Distribución de precios')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'area' in df.columns and len(df):
            fig2 = px.scatter(df, x='area', y='precio', hover_data=[
                              'nombre', 'tipo'], title='Precio vs Área')
            st.plotly_chart(fig2, use_container_width=True)

    # download filtered data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar CSV (filtrado)', data=csv,
                       file_name=f'{ds_choice.lower()}_filtered.csv', mime='text/csv')

elif mode == 'Mapa de oportunidades':
    st.header('Mapa de oportunidades')

    # --- AGREGAR RESUMEN DE BARRIOS ---
    summary = sales.groupby(
        sales['nombre'].astype(str).str.strip().str.lower()
    ).agg(
        count=('precio', 'count'),
        median_price=('precio', 'median'),
        percent_under=('oportunity_houses', 'mean')
    ).reset_index()

    summary['percent_under'] *= 100
    summary.rename(columns={'nombre': 'barrio_norm'}, inplace=True)

    # unir con geojson
    gdf['barrio_norm'] = gdf.get('nombre', gdf.get('name', '').strip().lower())
    merged = gdf.merge(summary, on='barrio_norm', how='left')

    merged[['percent_under', 'count', 'median_price']] = (
        merged[['percent_under', 'count', 'median_price']].fillna(0)
    )

    # --- MAPA CHOROPLETH ---
    st.subheader('Choropleth - porcentaje de listings subvalorados (venta)')
    fig = px.choropleth_mapbox(
        merged,
        geojson=merged.geometry.__geo_interface__,
        locations=merged.index,
        color='percent_under',
        mapbox_style='carto-positron',
        center=dict(
            lat=merged.geometry.centroid.y.mean(),
            lon=merged.geometry.centroid.x.mean()
        ),
        zoom=10,
        hover_data=['barrio_norm', 'count', 'median_price', 'percent_under']
    )
    fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

    st.plotly_chart(fig, use_container_width=True)

    # --- TABLA DE OPORTUNIDADES ---
    st.markdown('### Propiedades oportunidad')
    opp = sales[sales['oportunity_houses']].copy()

    cols_show = ["id", "nombre", "tipo", "precio", "area",
                 "habitaciones", "baños", "parqueaderos",
                 "latitud", "longitud"]

    cols_show = [c for c in cols_show if c in opp.columns]

    selected = st.data_editor(
        opp[cols_show],
        height=350,
        use_container_width=True,
        key="opp_selector"
    )

    # --- SI UNA PROPIEDAD ES SELECCIONADA: MOSTRAR MAPA ---
    if isinstance(selected, pd.DataFrame) and len(selected) == 1:
        prop = selected.iloc[0]

        st.subheader("📍 Ubicación de la propiedad seleccionada")

        if "latitud" in prop and "longitud" in prop:
            df_point = pd.DataFrame({
                "lat": [prop["latitud"]],
                "lon": [prop["longitud"]],
                "label": [f"{prop['tipo']} - ${prop['precio']:,.0f}"]
            })

            fig2 = px.scatter_mapbox(
                df_point,
                lat="lat",
                lon="lon",
                size=[30],
                hover_name="label",
                zoom=16,
                mapbox_style="carto-positron"
            )

            fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Esta propiedad no tiene latitud/longitud.")

    # --- DESCARGA ---
    csv2 = opp.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Descargar oportunidades (venta)',
        data=csv2,
        file_name='oportunidades_venta.csv',
        mime='text/csv'
    )



# -----------------------------
# Footer / tips
# -----------------------------
st.markdown('---')
st.caption('App creada por Roman Alejandro Correa.')
