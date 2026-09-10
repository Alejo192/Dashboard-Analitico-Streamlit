"""
app.py
======
Dashboard Analítico de Priorización de Leads Inmobiliarios
Construido con Streamlit y Plotly según especificaciones:
- Rol 01 Data Engineer: Lectura directa de data/db_dashboard_course.db (tbl_leads, dim_hobby, dim_comentario).
- Rol 02 Data Analyst: Priorización y orden_contacto vía logic_priorizacion.py.
- Rol 03 Data Scientist: Etiquetas de perfil e interpretación de bandas vía interpretacion.py.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic_priorizacion import (
    calcular_prioridad_y_orden,
    CLUSTER_TOP_CONVERSION,
    UMBRAL_ALTO,
    UMBRAL_MEDIO
)
from interpretacion import (
    enriquecer_con_interpretacion,
    ETIQUETAS_CLUSTER,
    INFO_RESOLUCION_DESACUERDO
)

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Priorización Comercial de Leads",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CARGA Y CACHÉ DE DATOS DIRECTO DE SQLite
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos_sqlite():
    """
    Lee directamente data/db_dashboard_course.db.
    No recalcula Cluster ni Probabilidad_Compra.
    Aplica las capas de negocio de Rol 2 (Priorización) y Rol 3 (Interpretación).
    """
    db_path = Path(__file__).resolve().parent / "data" / "db_dashboard_course.db"
    if not db_path.exists():
        st.error(f"Base de datos no encontrada en: {db_path}")
        st.stop()

    conn = sqlite3.connect(db_path)
    df_leads_raw = pd.read_sql_query("SELECT * FROM tbl_leads", conn)
    df_hobby = pd.read_sql_query("SELECT * FROM dim_hobby ORDER BY hobby_estandar", conn)
    df_comentario = pd.read_sql_query("SELECT * FROM dim_comentario ORDER BY comentario_id", conn)
    conn.close()

    # Aplicar lógica del Rol 2: Priorización y orden determinista
    df_priorizado = calcular_prioridad_y_orden(df_leads_raw)

    # Aplicar lógica del Rol 3: Guía de interpretación
    df_enriquecido = enriquecer_con_interpretacion(df_priorizado)

    return df_enriquecido, df_hobby, df_comentario


df_leads, df_hobby, df_comentario = cargar_datos_sqlite()

# -----------------------------------------------------------------------------
# BARRA LATERAL (FILTROS Y CATÁLOGOS)
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/real-estate.png", width=64)
st.sidebar.title("Filtros Comerciales")

# 1. Filtro por Tier de Prioridad
tiers_disponibles = ['Alto', 'Medio', 'Bajo']
selected_tiers = st.sidebar.multiselect(
    "Tier de Prioridad:",
    options=tiers_disponibles,
    default=tiers_disponibles
)

# 2. Filtro por Cluster
cluster_options = sorted(df_leads['Cluster'].unique())
selected_clusters = st.sidebar.multiselect(
    "Segmento (Cluster):",
    options=cluster_options,
    default=cluster_options,
    format_func=lambda c: f"Cluster {c}: {ETIQUETAS_CLUSTER.get(c, {}).get('nombre_corto', '')}"
)

# 3. Filtro por Segmento de Hobby (dim_hobby)
hobbies_disponibles = ["TODOS"] + sorted(df_hobby['hobby_estandar'].unique().tolist())
selected_hobby = st.sidebar.selectbox(
    "Interés / Hobby (dim_hobby):",
    options=hobbies_disponibles
)

# 4. Búsqueda directa por ID de Prospecto
search_id = st.sidebar.number_input("Buscar IDPROSPECTO exacto (0 = ver todos):", min_value=0, value=0, step=1)

# APLICAR FILTROS
df_filtered = df_leads.copy()

if selected_tiers:
    df_filtered = df_filtered[df_filtered['tier_prioridad'].isin(selected_tiers)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_clusters:
    df_filtered = df_filtered[df_filtered['Cluster'].isin(selected_clusters)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_hobby != "TODOS":
    df_filtered = df_filtered[df_filtered['Hobbies_Estandar'] == selected_hobby]

if search_id > 0:
    df_filtered = df_filtered[df_filtered['IDPROSPECTO'] == search_id]

# Asegurar orden determinista estricto (Paso 2)
df_filtered = df_filtered.sort_values(by='orden_contacto', ascending=True)

# Sección informativa de Lookup Tables en Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Taxonomía de Negocio")

with st.sidebar.expander("Catálogo Comentarios (dim_comentario)"):
    st.caption(
        "**Catálogo de vocabulario independiente (14 categorías)**. "
        "Sin unión fila a fila con leads (acuerdo cerrado en Rol 3)."
    )
    st.dataframe(
        df_comentario[['comentario_id', 'categoria_comentario']],
        use_container_width=True,
        hide_index=True
    )

with st.sidebar.expander("Resolución de Desacuerdo (Lookups)"):
    st.markdown(
        """
        - **`dim_hobby`**: Join foráneo íntegro con `tbl_leads.Hobbies_Estandar` (0 huérfanos). Apta como feature y filtro.
        - **`dim_comentario`**: Catálogo de referencia textual únicamente. No disponible como feature fila a fila.
        """
    )

# -----------------------------------------------------------------------------
# ENCABEZADO Y KPIs
# -----------------------------------------------------------------------------
st.title("🏢 Priorización Analítica de Leads Inmobiliarios")
st.markdown(
    """
    **Herramienta de Decisión Comercial:** scoring y ranking de prospectos para maximizar la efectividad del equipo de ventas.
    Datos congelados desde `data/db_dashboard_course.db` sin recalcular modelos.
    """
)

col1, col2, col3, col4 = st.columns(4)

total_leads_filtro = len(df_filtered)
total_leads_base = len(df_leads)
leads_altos_filtro = (df_filtered['tier_prioridad'] == 'Alto').sum()
leads_medios_filtro = (df_filtered['tier_prioridad'] == 'Medio').sum()
tasa_conv_media = df_filtered['Probabilidad_Compra'].mean() if total_leads_filtro > 0 else 0.0

col1.metric("Leads Seleccionados", f"{total_leads_filtro:,}", f"de {total_leads_base} en base")
col2.metric("Prioridad Alta", f"{leads_altos_filtro}", f"{leads_altos_filtro/total_leads_filtro*100:.1f}%" if total_leads_filtro > 0 else "0%")
col3.metric("Prioridad Media", f"{leads_medios_filtro}", f"{leads_medios_filtro/total_leads_filtro*100:.1f}%" if total_leads_filtro > 0 else "0%")
col4.metric("Probabilidad Media", f"{tasa_conv_media*100:.1f}%", "Conversión esperada")

st.markdown("---")

# -----------------------------------------------------------------------------
# VISUALIZACIONES ANALÍTICAS (PLOTLY)
# -----------------------------------------------------------------------------
st.subheader("📊 Análisis Visual de Segmentación y Prioridad")

tab_graficos1, tab_graficos2 = st.tabs(["Distribución de Prioridad por Cluster", "Distribución de Probabilidad de Compra"])

with tab_graficos1:
    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        # Gráfica de barras apiladas: Cluster vs Tier de Prioridad
        df_plot_cluster = df_leads.groupby(['Cluster', 'tier_prioridad']).size().reset_index(name='Conteo')
        df_plot_cluster['Cluster_Label'] = df_plot_cluster['Cluster'].apply(
            lambda c: f"Cluster {c}: {ETIQUETAS_CLUSTER.get(c, {}).get('nombre_corto', '')}"
        )
        
        color_map = {'Alto': '#2E7D32', 'Medio': '#F57C00', 'Bajo': '#D32F2F'}

        fig_cluster = px.bar(
            df_plot_cluster,
            x='Cluster_Label',
            y='Conteo',
            color='tier_prioridad',
            color_discrete_map=color_map,
            category_orders={'tier_prioridad': ['Alto', 'Medio', 'Bajo']},
            barmode='stack',
            title="Volumen de Prospectos por Cluster y Tier de Prioridad",
            labels={'Cluster_Label': 'Segmento de Cluster', 'Conteo': 'Cantidad de Leads', 'tier_prioridad': 'Prioridad'}
        )
        fig_cluster.update_layout(height=380, legend_title_text='Tier Prioridad')
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col_g2:
        st.markdown("#### Interpretación de Clusters (Rol Data Scientist)")
        for c in sorted(ETIQUETAS_CLUSTER.keys()):
            info = ETIQUETAS_CLUSTER[c]
            st.markdown(f"**{info['etiqueta_completa']}**")
            st.caption(f"• **Perfil:** {info['descripcion']}")
            st.caption(f"• **Acción Comercial:** {info['recomendacion_comercial']}")
            st.write("")

with tab_graficos2:
    # Histograma con umbrales de decisión
    fig_prob = px.histogram(
        df_leads,
        x='Probabilidad_Compra',
        color='tier_prioridad',
        color_discrete_map=color_map,
        category_orders={'tier_prioridad': ['Alto', 'Medio', 'Bajo']},
        nbins=25,
        marginal='box',
        title="Histograma de Probabilidad_Compra con Umbrales de Negocio (0.40 y 0.70)",
        labels={'Probabilidad_Compra': 'Probabilidad de Compra', 'count': 'Frecuencia'}
    )
    fig_prob.add_vline(x=UMBRAL_MEDIO, line_dash="dash", line_color="#F57C00", annotation_text="Corte Medio (0.40)")
    fig_prob.add_vline(x=UMBRAL_ALTO, line_dash="dash", line_color="#2E7D32", annotation_text="Corte Alto (0.70)")
    fig_prob.update_layout(height=420)
    st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# TABLA PRINCIPAL DE CONTACTO COMERCIAL
# -----------------------------------------------------------------------------
st.subheader("📞 Hoja de Ruta Comercial: Lista de Prospectos Priorizada")
st.caption("Ordenada deterministamente por 'orden_contacto' (1 = llamar hoy mismo).")

# Formatear columnas para visualización amigable
cols_display = [
    'orden_contacto',
    'tier_prioridad',
    'IDPROSPECTO',
    'Probabilidad_Compra',
    'banda_probabilidad',
    'cluster_nombre_corto',
    'Hobbies_Estandar',
    'Edad',
    'Salario_MarcaClase',
    'InteresMetraje_MarcaClase',
    'ApartamentoDesde_MarcaClase',
    'Compra'
]

df_table = df_filtered[cols_display].copy()

# Formatear probabilidad a porcentaje
df_table['Probabilidad_Compra'] = df_table['Probabilidad_Compra'].apply(lambda x: f"{x*100:.1f}%")
df_table['Salario_MarcaClase'] = df_table['Salario_MarcaClase'].apply(lambda x: f"${x:,.0f}")
df_table['Edad'] = df_table['Edad'].apply(lambda x: f"{x:.0f} años" if pd.notna(x) else "N/A")

st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        'orden_contacto': st.column_config.NumberColumn("Orden Llamada #", format="%d"),
        'tier_prioridad': st.column_config.TextColumn("Tier Prioridad"),
        'IDPROSPECTO': st.column_config.NumberColumn("ID Lead", format="%d"),
        'Probabilidad_Compra': st.column_config.TextColumn("Prob. Compra"),
        'banda_probabilidad': st.column_config.TextColumn("Banda Confianza"),
        'cluster_nombre_corto': st.column_config.TextColumn("Perfil Cluster"),
        'Hobbies_Estandar': st.column_config.TextColumn("Hobby / Interés"),
        'InteresMetraje_MarcaClase': st.column_config.NumberColumn("Metraje Interés", format="%d m²"),
        'Compra': st.column_config.TextColumn("Segmento de Valor (Compra)")
    }
)

# -----------------------------------------------------------------------------
# VERIFICACIÓN DE PROSPECTOS TESTIGO (AUTOVALIDACIÓN INTEGRADA)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Inspección y Verificación de Prospectos Clave (Paso 2 vs Dashboard)")

col_testigo1, col_testigo2, col_testigo3 = st.columns(3)

for col, id_p in zip([col_testigo1, col_testigo2, col_testigo3], [4, 8, 2]):
    lead_row = df_leads[df_leads['IDPROSPECTO'] == id_p]
    if not lead_row.empty:
        r = lead_row.iloc[0]
        tier = r['tier_prioridad']
        color_box = "🟢" if tier == "Alto" else ("🟠" if tier == "Medio" else "🔴")
        with col:
            st.markdown(f"#### {color_box} Prospecto ID: {id_p}")
            st.markdown(f"- **Tier**: `{tier}`")
            st.markdown(f"- **Orden de Contacto**: `#{r['orden_contacto']}`")
            st.markdown(f"- **Probabilidad**: `{r['Probabilidad_Compra']*100:.1f}%` ({r['banda_probabilidad']})")
            st.markdown(f"- **Cluster**: `{r['Cluster']}` ({r['cluster_nombre_corto']})")
            st.markdown(f"- **Hobby**: `{r['Hobbies_Estandar']}`")
            st.markdown(f"- **Segmento Compra**: `{r['Compra']}`")
