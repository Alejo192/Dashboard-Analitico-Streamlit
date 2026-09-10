"""
logic_priorizacion.py
=====================
Implementación de las reglas de negocio del Rol 02 (Analista de Datos)
según roles/02-data-analyst.md.

Decisiones analíticas tomadas con base en evidencia (documentadas explícitamente):
1. Cluster de mayor conversión histórica:
   - Calculado agrupando tbl_leads por 'Cluster' y extrayendo la media de 'Probabilidad_Compra'.
   - Cluster 2 tiene la mayor conversión media (0.5815 frente a 0.1848 de Cluster 0 y 0.1560 de Cluster 1).
   - Constante definida: CLUSTER_TOP_CONVERSION = 2.

2. Umbrales de Priorización comercial:
   - UMBRAL_ALTO = 0.70:
     Requiere pertenecer a CLUSTER_TOP_CONVERSION (Cluster 2) Y tener Probabilidad_Compra >= 0.70.
     Clasifica a 184 prospectos (~29.6% de la base), representando el núcleo de atención inmediata.
   - UMBRAL_MEDIO = 0.40 (Decisión propia del analista con evidencia empírica):
     El spec no define el corte para Medio vs Bajo.
     En el subconjunto de prospectos que no califican como Alto (n=438), la mediana de probabilidad es 0.055
     y el 65.8% tiene probabilidad <= 0.10. Un umbral de 0.40 selecciona al percentil 90 del remanente (51 leads),
     incluyendo prospectos con alta probabilidad pero de clusters 0 y 1, así como prospectos moderados de Cluster 2.
   - Resto (< 0.40):
     Clasificado como 'Bajo' (387 leads, ~62.2%), prospectos fríos o de maduración a largo plazo.

3. Desempate y Orden de Contacto (orden_contacto):
   - Orden prioritario determinista:
     1. tier_prioridad ('Alto' primero, luego 'Medio', luego 'Bajo').
     2. Probabilidad_Compra descendente.
     3. IDPROSPECTO ascendente (criterio técnico determinista ante empates).
   - Hobbies_Estandar / dim_hobby NUNCA altera el orden numérico de contacto; se reserva como dimensión visual.
   - La columna 'Compra' (texto categórico de valor) NO se usa como señal de conversión.
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Constantes de negocio determinadas con evidencia empírica
CLUSTER_TOP_CONVERSION = 2
UMBRAL_ALTO = 0.70
UMBRAL_MEDIO = 0.40

# Mapeo de ranking de tier
TIER_ORDER = {
    'Alto': 1,
    'Medio': 2,
    'Bajo': 3
}


def calcular_promedios_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa tbl_leads por Cluster y calcula estadísticas descriptivas de Probabilidad_Compra
    para evidenciar cuál es el cluster de mayor conversión histórica.
    """
    stats = df.groupby('Cluster')['Probabilidad_Compra'].agg(
        count='count',
        media='mean',
        mediana='median',
        std='std',
        minimo='min',
        maximo='max'
    ).reset_index()
    return stats


def asignar_tier_prioridad(row) -> str:
    """
    Asigna el tier de prioridad (Alto / Medio / Bajo) a un lead individual
    con base en Cluster y Probabilidad_Compra.
    """
    prob = row['Probabilidad_Compra']
    cluster = row['Cluster']

    if pd.isna(prob):
        raise ValueError(f"Probabilidad_Compra no puede ser nula para el lead {row.get('IDPROSPECTO')}")

    # Condición de Prioridad Alta
    if cluster == CLUSTER_TOP_CONVERSION and prob >= UMBRAL_ALTO:
        return 'Alto'
    # Condición de Prioridad Media (decisión de negocio documentada)
    elif prob >= UMBRAL_MEDIO:
        return 'Medio'
    # Condición de Prioridad Baja
    else:
        return 'Bajo'


def calcular_prioridad_y_orden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la lógica completa de priorización y desempate:
    1. Calcula la columna 'tier_prioridad' ('Alto', 'Medio', 'Bajo').
    2. Ordena por:
       - tier_prioridad (Alto -> Medio -> Bajo)
       - Probabilidad_Compra descendente
       - IDPROSPECTO ascendente (desempate determinista)
    3. Asigna 'orden_contacto' (entero 1..N indicando la secuencia de llamada).
    
    Retorna una copia del DataFrame ordenado con ambas columnas agregadas.
    """
    df_out = df.copy()

    # 1. Asignar tier de prioridad
    df_out['tier_prioridad'] = df_out.apply(asignar_tier_prioridad, axis=1)

    # 2. Asignar peso numérico temporal para ordenamiento
    df_out['tier_rank'] = df_out['tier_prioridad'].map(TIER_ORDER)

    # 3. Orden determinista
    # Nota: Probabilidad_Compra con tolerancia / redondeo implícito por formato float
    df_out = df_out.sort_values(
        by=['tier_rank', 'Probabilidad_Compra', 'IDPROSPECTO'],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    # 4. Asignar orden de contacto (1 = primer contacto comercial)
    df_out['orden_contacto'] = df_out.index + 1

    # Eliminar columna auxiliar de orden
    df_out = df_out.drop(columns=['tier_rank'])

    return df_out


def ejecutar_demostracion(db_path: str = None):
    """
    Ejecuta el cálculo contra la base de datos real y presenta:
    1. La tabla de promedios de conversión por Cluster.
    2. La distribución de leads por Tier de Prioridad.
    3. El resultado específico para 3 prospectos testigo (Alto, Medio, Bajo).
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parent / 'data' / 'db_dashboard_course.db'

    conn = sqlite3.connect(db_path)
    df_leads = pd.read_sql_query("SELECT * FROM tbl_leads", conn)
    conn.close()

    print("=" * 70)
    print("PASO 2 — EVIDENCIA: PROMEDIO DE PROBABILIDAD DE COMPRA POR CLUSTER")
    print("=" * 70)
    cluster_stats = calcular_promedios_cluster(df_leads)
    print(cluster_stats.to_string(index=False))

    print("\n" + "=" * 70)
    print("CALCULANDO PRIORIZACIÓN Y ORDEN DE CONTACTO...")
    print("=" * 70)
    df_priorizado = calcular_prioridad_y_orden(df_leads)

    print("\nDistribución resultante por Tier de Prioridad:")
    print(df_priorizado['tier_prioridad'].value_counts().to_string())

    print("\n" + "=" * 70)
    print("RESULTADO DE 3 PROSPECTOS REALES DE TESTIGO (ALTO, MEDIO, BAJO)")
    print("=" * 70)
    
    # Seleccionamos:
    # 1. Lead Alto: IDPROSPECTO 4 (Cluster 2, Prob 1.0)
    # 2. Lead Medio: IDPROSPECTO 8 (Cluster 2, Prob 0.46)
    # 3. Lead Bajo: IDPROSPECTO 2 (Cluster 0, Prob 0.02)
    ids_testigo = [4, 8, 2]
    cols_mostrar = ['orden_contacto', 'IDPROSPECTO', 'Cluster', 'Probabilidad_Compra', 'Hobbies_Estandar', 'Compra', 'tier_prioridad']
    
    df_testigo = df_priorizado[df_priorizado['IDPROSPECTO'].isin(ids_testigo)][cols_mostrar].sort_values(by='orden_contacto')
    print(df_testigo.to_string(index=False))

    return df_priorizado, df_testigo


if __name__ == '__main__':
    ejecutar_demostracion()
