"""
Método de Hardy Cross para redes cerradas de tuberías
Ecuación de pérdidas: Hazen-Williams (unidades SI)

Ejecutar con:   streamlit run hardy_cross_app.py

Resumen del método (para explicarlo paso a paso):
  1. Se supone un caudal inicial Q en cada tramo que cumpla la continuidad en los nodos.
  2. Para cada tramo se calcula la resistencia K de Hazen-Williams:
         K = 10.67 · L / (C^1.852 · D^4.87)      ->  hf = K · Q^1.852
  3. En cada circuito se suma la pérdida de carga con signo (+ horario, - antihorario).
     Si la red está balanceada, esa suma es cero.
  4. Se calcula la corrección de caudal del circuito:
         ΔQ = - Σ(hf con signo) / (1.852 · Σ|hf / Q|)
  5. Se corrige cada tramo: Q_nuevo = Q + ΔQ  (los tramos compartidos reciben
     la corrección de los dos circuitos vecinos, cada una con su propio signo).
  6. Se repite hasta que max|ΔQ| < tolerancia.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constantes y configuración
# ----------------------------------------------------------------------------
EXPONENTE = 1.852                      # Exponente de Q en Hazen-Williams
COLS_CIRCUITO = ["Circ_1", "Circ_2", "Circ_3", "Circ_4"]   # Máx. 4 circuitos
COLS_DATOS = ["Tramo", "L_m", "D_m", "C_HW", "Q0_m3s"] + COLS_CIRCUITO

st.set_page_config(page_title="Hardy Cross – Hazen-Williams", layout="wide")


# ----------------------------------------------------------------------------
# Red de ejemplo (2 circuitos)
# ----------------------------------------------------------------------------
def red_ejemplo() -> pd.DataFrame:
    """
    Dos circuitos cuadrados que comparten el tramo B-E:

            A ---1--- B ---5--- C          Circuito 1: A-B-E-F  (horario)
            |         |         |          Circuito 2: B-C-D-E  (horario)
            4         2         6
            |         |         |          Entra 0.30 m³/s en A
            F ---3--- E ---7--- D          Sale  0.30 m³/s en D

    Q0 es SIEMPRE positivo y va en el sentido en que se nombra el tramo
    (por ejemplo "A-B" va de A hacia B).
    Circ_k vale +1 si ese sentido coincide con el giro horario del circuito k,
    -1 si es antihorario, y 0 si el tramo no pertenece al circuito.
    """
    datos = [
        # Tramo, L(m),  D(m),  C,   Q0,   C1,  C2,  C3, C4
        ["A-B", 600.0, 0.30, 120, 0.18, +1, 0, 0, 0],
        ["B-E", 500.0, 0.25, 120, 0.08, +1, -1, 0, 0],   # tramo compartido
        ["F-E", 600.0, 0.25, 120, 0.12, -1, 0, 0, 0],
        ["A-F", 500.0, 0.30, 120, 0.12, -1, 0, 0, 0],
        ["B-C", 500.0, 0.25, 120, 0.10, 0, +1, 0, 0],
        ["C-D", 600.0, 0.25, 120, 0.10, 0, +1, 0, 0],
        ["E-D", 500.0, 0.30, 120, 0.20, 0, -1, 0, 0],
    ]
    return pd.DataFrame(datos, columns=COLS_DATOS)


# ----------------------------------------------------------------------------
# Funciones de cálculo
# ----------------------------------------------------------------------------
def resistencia_k(L, D, C):
    """Resistencia de la tubería: K = 10.67 L / (C^1.852 D^4.87)."""
    return 10.67 * L / (C ** EXPONENTE * D ** 4.87)


def validar(df: pd.DataFrame) -> list[str]:
    """Devuelve una lista de errores en la tabla (vacía si todo está bien)."""
    errores = []
    if df.empty:
        return ["La tabla está vacía."]
    if df[["L_m", "D_m", "C_HW", "Q0_m3s"] + COLS_CIRCUITO].isna().any().any():
        errores.append("Hay celdas vacías en la tabla; complétalas o elimina la fila.")
        return errores
    if (df["L_m"] <= 0).any() or (df["D_m"] <= 0).any() or (df["C_HW"] <= 0).any():
        errores.append("L, D y C deben ser mayores que cero.")
    if (df["Q0_m3s"] <= 0).any():
        errores.append("Q0 debe ser positivo; el sentido se define con las columnas Circ_k.")
    if not df[COLS_CIRCUITO].isin([-1, 0, 1]).all().all():
        errores.append("Las columnas Circ_k solo admiten los valores -1, 0 o 1.")
    if (df[COLS_CIRCUITO].abs().sum(axis=1) == 0).any():
        errores.append("Hay tramos que no pertenecen a ningún circuito (todos sus Circ_k valen 0).")
    if (df[COLS_CIRCUITO].abs().sum(axis=0) == 0).all():
        errores.append("Ningún circuito tiene tramos asignados.")
    return errores


def hardy_cross(df: pd.DataFrame, tol: float, max_iter: int) -> dict:
    """
    Ejecuta Hardy Cross sobre la tabla de tramos.

    Devuelve un diccionario con:
      - iteraciones : DataFrame con ΔQ, cierre ΣhL y error por iteración
      - caudales    : DataFrame con el caudal de cada tramo en cada iteración
      - Q_final     : arreglo con los caudales finales (con signo)
      - K           : resistencias de cada tramo
      - convergio   : True si se alcanzó la tolerancia
    """
    L = df["L_m"].to_numpy(float)
    D = df["D_m"].to_numpy(float)
    C = df["C_HW"].to_numpy(float)
    Q = df["Q0_m3s"].to_numpy(float).copy()

    # Matriz de signos S (n_tramos × n_circuitos) y sus valores absolutos
    S = df[COLS_CIRCUITO].to_numpy(float)
    S_abs = np.abs(S)

    K = resistencia_k(L, D, C)

    filas_iter = []                                  # Resumen por iteración
    hist_Q = [pd.Series(Q, index=df["Tramo"], name="Iteración 0")]
    convergio = False

    for it in range(1, max_iter + 1):
        # Pérdida con signo de cada tramo: hf = K · Q · |Q|^0.852
        hf = K * Q * np.abs(Q) ** (EXPONENTE - 1)

        # Numerador por circuito: Σ(s · hf)  -> error de cierre del circuito
        cierre = S.T @ hf

        # Denominador por circuito: 1.852 · Σ|hf/Q| = 1.852 · Σ K |Q|^0.852
        den = EXPONENTE * (S_abs.T @ (K * np.abs(Q) ** (EXPONENTE - 1)))

        # Corrección de caudal de cada circuito (si el circuito está vacío, ΔQ = 0)
        dQ = np.divide(-cierre, den, out=np.zeros_like(cierre), where=den > 0)

        # Se aplican todas las correcciones a la vez; un tramo compartido
        # recibe la corrección de ambos circuitos, cada una con su signo.
        Q = Q + S @ dQ

        error = np.max(np.abs(dQ))
        fila = {"Iteración": it}
        for k, nombre in enumerate(COLS_CIRCUITO):
            if S_abs[:, k].sum() > 0:                # Solo circuitos usados
                fila[f"ΣhL {nombre} (m)"] = cierre[k]
                fila[f"ΔQ {nombre} (m³/s)"] = dQ[k]
        fila["Error máx |ΔQ|"] = error
        filas_iter.append(fila)
        hist_Q.append(pd.Series(Q, index=df["Tramo"], name=f"Iteración {it}"))

        if error < tol:
            convergio = True
            break

    return {
        "iteraciones": pd.DataFrame(filas_iter),
        "caudales": pd.concat(hist_Q, axis=1).T,
        "Q_final": Q,
        "K": K,
        "convergio": convergio,
    }


def tabla_resultados(df: pd.DataFrame, res: dict) -> pd.DataFrame:
    """Arma la tabla final: caudal, sentido real, velocidad y pérdidas."""
    Q = res["Q_final"]
    K = res["K"]
    area = np.pi * df["D_m"].to_numpy(float) ** 2 / 4
    hf = K * np.abs(Q) ** EXPONENTE

    return pd.DataFrame({
        "Tramo": df["Tramo"].to_numpy(),
        "L (m)": df["L_m"].to_numpy(),
        "D (m)": df["D_m"].to_numpy(),
        "C": df["C_HW"].to_numpy(),
        "Q inicial (m³/s)": df["Q0_m3s"].to_numpy(),
        "Q final (m³/s)": np.abs(Q),
        # Si el caudal final quedó negativo, el flujo va contra el sentido supuesto
        "Sentido real": np.where(Q >= 0, "Como se supuso", "Invertido"),
        "V (m/s)": np.abs(Q) / area,
        "hf (m)": hf,
        "J (m/m)": hf / df["L_m"].to_numpy(),
    })


def graficar_convergencia(it_df: pd.DataFrame, tol: float):
    """Gráfica de ΔQ por iteración (escala lineal y escala logarítmica)."""
    cols_dq = [c for c in it_df.columns if c.startswith("ΔQ ")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    for c in cols_dq:
        nombre = c.split(" ")[1]                          # p. ej. "Circ_1"
        ax1.plot(it_df["Iteración"], it_df[c], marker="o", label=nombre)
        # En escala log se grafica |ΔQ| (se evita el cero para no romper el log)
        ax2.semilogy(it_df["Iteración"], np.maximum(it_df[c].abs(), 1e-12),
                     marker="o", label=nombre)

    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_title("ΔQ por iteración (con signo)")
    ax1.set_ylabel("ΔQ (m³/s)")

    ax2.axhline(tol, color="red", ls="--", lw=1, label=f"Tolerancia = {tol:g}")
    ax2.set_title("|ΔQ| por iteración (escala log)")
    ax2.set_ylabel("|ΔQ| (m³/s)")

    for ax in (ax1, ax2):
        ax.set_xlabel("Iteración")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


def exportar_excel(df, resultados, res, tol) -> bytes:
    """Genera el archivo .xlsx en memoria con todas las hojas de resultados."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        df.to_excel(escritor, sheet_name="Datos", index=False)
        resultados.to_excel(escritor, sheet_name="Resultados", index=False)
        res["iteraciones"].to_excel(escritor, sheet_name="Iteraciones", index=False)
        res["caudales"].to_excel(escritor, sheet_name="Caudales por iteración")
        pd.DataFrame({
            "Parámetro": ["Tolerancia", "Iteraciones realizadas", "¿Convergió?"],
            "Valor": [tol, len(res["iteraciones"]), "Sí" if res["convergio"] else "No"],
        }).to_excel(escritor, sheet_name="Resumen", index=False)
    return buffer.getvalue()


# ----------------------------------------------------------------------------
# Interfaz de Streamlit
# ----------------------------------------------------------------------------
st.title("Hardy Cross – Redes cerradas de tuberías")
st.caption("Ecuación de Hazen-Williams (SI):  hf = 10.67 · L · Q^1.852 / (C^1.852 · D^4.87)")

# La red se guarda en session_state para conservar las ediciones del usuario
if "red" not in st.session_state:
    st.session_state["red"] = red_ejemplo()

# --- Barra lateral: parámetros del cálculo ---
with st.sidebar:
    st.header("Parámetros")
    tol = st.number_input("Tolerancia de ΔQ (m³/s)", min_value=1e-9, value=0.0001,
                          step=0.0001, format="%.6f")
    max_iter = st.number_input("Máximo de iteraciones", min_value=1, max_value=1000,
                               value=100, step=10)
    if st.button("Restablecer red de ejemplo"):
        st.session_state["red"] = red_ejemplo()
        st.session_state.pop("editor", None)     # Descarta ediciones pendientes
        st.rerun()

    st.markdown(
        "**Cómo llenar la tabla**\n\n"
        "- `Q0_m3s` siempre positivo, en el sentido del nombre del tramo.\n"
        "- `Circ_k`: **+1** si ese sentido es horario en el circuito k, "
        "**-1** si es antihorario, **0** si no pertenece.\n"
        "- Un tramo compartido tiene valores distintos de 0 en dos circuitos.\n"
        "- Los Q0 deben cumplir la continuidad en cada nodo."
    )

# --- Tabla editable ---
st.subheader("1. Datos de la red")
df_editado = st.data_editor(
    st.session_state["red"],
    key="editor",
    num_rows="dynamic",                  # Permite agregar y borrar tramos
    use_container_width=True,
    column_config={
        "Tramo": st.column_config.TextColumn("Tramo"),
        "L_m": st.column_config.NumberColumn("Longitud L (m)", min_value=0.0, format="%.1f"),
        "D_m": st.column_config.NumberColumn("Diámetro D (m)", min_value=0.0, format="%.3f"),
        "C_HW": st.column_config.NumberColumn("Coef. C", min_value=0.0, format="%.0f"),
        "Q0_m3s": st.column_config.NumberColumn("Q inicial (m³/s)", min_value=0.0, format="%.4f"),
        **{c: st.column_config.NumberColumn(c, min_value=-1, max_value=1, step=1, format="%d")
           for c in COLS_CIRCUITO},
    },
)

# Limpieza: nombres vacíos reciben un nombre automático y se descartan filas vacías
df = df_editado.dropna(how="all").reset_index(drop=True)
df["Tramo"] = [t if isinstance(t, str) and t.strip() else f"T{i + 1}"
               for i, t in enumerate(df["Tramo"])]

errores = validar(df)
if errores:
    for e in errores:
        st.error(e)
    st.stop()

# --- Cálculo automático ---
res = hardy_cross(df, tol, int(max_iter))
resultados = tabla_resultados(df, res)
it_df = res["iteraciones"]

if res["convergio"]:
    st.success(f"Convergió en {len(it_df)} iteraciones "
               f"(error final = {it_df['Error máx |ΔQ|'].iloc[-1]:.2e} m³/s).")
else:
    st.warning(f"No convergió en {len(it_df)} iteraciones. "
               "Revisa los datos o aumenta el máximo de iteraciones.")

# --- Iteraciones ---
st.subheader("2. Iteraciones y ajuste ΔQ")
st.dataframe(it_df.style.format({c: "{:.6f}" for c in it_df.columns if c != "Iteración"}),
             use_container_width=True, hide_index=True)
with st.expander("Ver caudales de cada tramo en cada iteración"):
    st.dataframe(res["caudales"].style.format("{:.5f}"), use_container_width=True)

# --- Gráfica ---
st.subheader("3. Convergencia de ΔQ")
fig = graficar_convergencia(it_df, tol)
st.pyplot(fig)

# --- Resultados finales ---
st.subheader("4. Resultados finales")
st.dataframe(
    resultados.style.format({
        "L (m)": "{:.1f}", "D (m)": "{:.3f}", "C": "{:.0f}",
        "Q inicial (m³/s)": "{:.5f}", "Q final (m³/s)": "{:.5f}",
        "V (m/s)": "{:.3f}", "hf (m)": "{:.3f}", "J (m/m)": "{:.5f}",
    }),
    use_container_width=True, hide_index=True,
)

# --- Descarga ---
st.download_button(
    "Descargar resultados en Excel",
    data=exportar_excel(df, resultados, res, tol),
    file_name="hardy_cross_resultados.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
