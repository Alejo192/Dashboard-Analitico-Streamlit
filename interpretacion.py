"""
interpretacion.py
=================
Implementación de la guía de interpretación para el Rol 03 (Científico de Datos)
según roles/03-data-scientist.md.

NOTA METODOLÓGICA IMPORTANTE:
El spec menciona que la etiqueta de cada Cluster depende del análisis de perfil
realizado originalmente en el Módulo 4 (DataScienceAplicado-Fundamentals).
Al no leerse dicho repositorio externo en este protocolo, las etiquetas aquí
definidas corresponden a una INTERPRETACIÓN PROPIA basada en el perfilado
empírico directo de 'tbl_leads' (edad, salario, metraje, ticket y probabilidad).
Esta interpretación debe ser revisada y validada posteriormente por un científico
de datos de dominio.

RESOLUCIÓN DEL DESACUERDO (Negociación entre Data Engineer y Data Scientist):
- 'dim_comentario': Se ratifica como tabla de vocabulario/catálogo de referencia únicamente.
  No posee clave de unión ni correspondencia fila a fila con 'tbl_leads'. Por tanto,
  NO se une a nivel de registro ni se utiliza como feature de modelo en esta iteración.
- 'dim_hobby': Posee un join documentado y verificado (0 huérfanos) con 'tbl_leads.Hobbies_Estandar'.
  Es apta para segmentación analítica y como feature categórica en futuras iteraciones.
"""

import pandas as pd
from typing import Dict, Any

# ==============================================================================
# 1. ETIQUETAS DE NEGOCIO PARA CLUSTERS (Perfilado propio sobre tbl_leads)
# ==============================================================================
# Evidencia empírica de tbl_leads:
# - Cluster 0 (n=29, 4.7%):
#     Edad media: 49.3 años | Salario medio: $3,008,621 (el más alto)
#     Ticket vendido: $236M (el más alto) | Metraje: 51.4 m2
#     Probabilidad de compra media: 0.185 (mediana 0.09)
#     Interpretación: Segmento minoritario de alto poder adquisitivo y propiedades de mayor valor,
#                     pero con baja urgencia / inmediatez de compra.
#
# - Cluster 1 (n=250, 40.2%):
#     Edad media: 63.4 años (rango 57.2 a 69.5 años; estrictamente adultos mayores/seniors)
#     Salario medio: $2,412,000 | Metraje interés: 56.4 m2 (el mayor metraje)
#     Ticket vendido: $186.8M | Probabilidad de compra media: 0.156 (mediana 0.02)
#     Interpretación: Adultos mayores/jubilados que buscan inmuebles amplios pero con
#                     la tasa más baja de conversión inmediata.
#
# - Cluster 2 (n=343, 55.1%):
#     Edad media: 47.7 años (mediana 45.9 años) | Salario medio: $2,317,055
#     Metraje interés: 39.2 m2 (más compacto) | Ticket vendido: $154.5M
#     Probabilidad de compra media: 0.581 (mediana 0.83; >50% con prob > 80%)
#     Interpretación: Núcleo dinámico de alta conversión y mayor volumen de ventas cerradas
#                     (31.8% high value contracts).
# ==============================================================================

ETIQUETAS_CLUSTER: Dict[int, Dict[str, str]] = {
    0: {
        'nombre_corto': 'Nicho Premium / Alto Poder Adquisitivo',
        'etiqueta_completa': 'Cluster 0: Perfil Premium / Alto Ingreso (Baja Inmediatez Comercial)',
        'descripcion': 'Clientes con el mayor ingreso medio ($3.0M) y tickets altos ($236M), pero con ciclo de decisión largo y baja probabilidad inmediata (media: 18.5%).',
        'recomendacion_comercial': 'Enfoque en propiedades exclusivas y seguimiento consultivo sin presión.'
    },
    1: {
        'nombre_corto': 'Senior / Preferencia Metraje Amplio',
        'etiqueta_completa': 'Cluster 1: Adultos Mayores / Inmuebles Amplios (Baja Conversión Inmediata)',
        'descripcion': 'Grupo homogéneo de adultos mayores (57-70 años) interesados en metrajes generosos (56.4 m2), pero con baja conversión inmediata (media: 15.6%, mediana: 2%).',
        'recomendacion_comercial': 'Presentar opciones accesibles, de fácil movilidad y proyectos con áreas sociales tranquilas.'
    },
    2: {
        'nombre_corto': 'Core Comercial / Alta Conversión',
        'etiqueta_completa': 'Cluster 2: Núcleo Comercial Dinámico (Alta Conversión y Decisión de Compra)',
        'descripcion': 'Segmento principal (55% de la base) con la mayor tasa de conversión histórica (media: 58.1%, mediana: 83.0%) y búsqueda de metrajes prácticos (39.2 m2).',
        'recomendacion_comercial': 'Prioridad máxima de contacto telefónico y agendamiento inmediato de visitas.'
    }
}


# ==============================================================================
# 2. BANDAS DE CONFIANZA DE PROBABILIDAD DE COMPRA
# ==============================================================================
# Consistente 100% con los umbrales definidos en el Paso 2 (Rol Data Analyst):
# - Alta: >= 0.70
# - Media: 0.40 - 0.69
# - Baja: < 0.40
# ==============================================================================

def clasificar_banda_probabilidad(prob: float) -> str:
    """
    Retorna la banda de confianza legible para un asesor comercial.
    """
    if pd.isna(prob):
        return 'No Definida'
    if prob >= 0.70:
        return 'Alta (>= 70%)'
    elif prob >= 0.40:
        return 'Media (40% - 69%)'
    else:
        return 'Baja (< 40%)'


def obtener_etiqueta_cluster(cluster_id: int, formato: str = 'etiqueta_completa') -> str:
    """
    Retorna la etiqueta legible del cluster según el formato solicitado
    ('nombre_corto', 'etiqueta_completa', 'descripcion', 'recomendacion_comercial').
    """
    cluster_info = ETIQUETAS_CLUSTER.get(cluster_id)
    if cluster_info:
        return cluster_info.get(formato, f'Cluster {cluster_id}')
    return f'Cluster Desconocido ({cluster_id})'


def enriquecer_con_interpretacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a un DataFrame que contenga 'Cluster' y 'Probabilidad_Compra':
    - 'cluster_etiqueta': Nombre legible del cluster.
    - 'cluster_nombre_corto': Nombre abreviado del cluster.
    - 'banda_probabilidad': Rango de confianza de la probabilidad.
    """
    df_out = df.copy()
    
    if 'Cluster' in df_out.columns:
        df_out['cluster_etiqueta'] = df_out['Cluster'].apply(
            lambda c: obtener_etiqueta_cluster(c, 'etiqueta_completa')
        )
        df_out['cluster_nombre_corto'] = df_out['Cluster'].apply(
            lambda c: obtener_etiqueta_cluster(c, 'nombre_corto')
        )

    if 'Probabilidad_Compra' in df_out.columns:
        df_out['banda_probabilidad'] = df_out['Probabilidad_Compra'].apply(
            clasificar_banda_probabilidad
        )

    return df_out


# ==============================================================================
# 3. METADATA DE RESOLUCIÓN DE DESACUERDO (Contrato entre roles)
# ==============================================================================

INFO_RESOLUCION_DESACUERDO: Dict[str, Any] = {
    'tabla_dim_hobby': {
        'tipo': 'Dimensión con clave foránea validada',
        'clave_join': 'tbl_leads.Hobbies_Estandar = dim_hobby.hobby_estandar',
        'orfanadad_verificada': True,
        'filas_huerfanas': 0,
        'apta_para_feature_modelo': True,
        'uso_dashboard': 'Filtro interactivo y segmentación por interés del prospecto.'
    },
    'tabla_dim_comentario': {
        'tipo': 'Catálogo de vocabulario / taxonomía independiente',
        'clave_join': None,
        'orfanadad_verificada': False,
        'filas_huerfanas': 'No aplica (sin relación de join)',
        'apta_para_feature_modelo': False,
        'uso_dashboard': 'Catálogo de referencia de categorías para el comercial (no se une a leads).'
    }
}


if __name__ == '__main__':
    print("=" * 70)
    print("PASO 3 — GUÍA DE INTERPRETACIÓN (ROL DATA SCIENTIST)")
    print("=" * 70)
    for c, info in ETIQUETAS_CLUSTER.items():
        print(f"\n[Cluster {c}]")
        print(f"  Etiqueta: {info['etiqueta_completa']}")
        print(f"  Nombre corto: {info['nombre_corto']}")
        print(f"  Descripción: {info['descripcion']}")
        print(f"  Recomendación: {info['recomendacion_comercial']}")

    print("\n" + "=" * 70)
    print("RESOLUCIÓN DEL DESACUERDO (LOOKUP TABLES):")
    print("=" * 70)
    for tabla, det in INFO_RESOLUCION_DESACUERDO.items():
        print(f"* {tabla}: {det['tipo']} -> Apta feature: {det['apta_para_feature_modelo']}")
