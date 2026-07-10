"""
Frontera Eficiente de Markowitz con datos reales
=================================================
Descarga 5 años de precios de 5 activos, estima retorno esperado y matriz de
varianzas-covarianzas, simula 10.000 carteras aleatorias y grafica la frontera
eficiente, identificando el portfolio de máximo Sharpe y el de mínima varianza.

Genera:
  - datos_precios.csv      -> cache de precios (reproducibilidad)
  - frontera_eficiente.png -> gráfico principal (frontera + portfolios óptimos)
  - ponderaciones.png      -> comparación de pesos de ambos portfolios
  - correlaciones.png      -> matriz de correlación entre activos

Uso:  python frontera_eficiente.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda archivos PNG
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# PARÁMETROS
# ----------------------------------------------------------------------------
# Activos elegidos buscando BAJA CORRELACIÓN entre clases:
#   SPY  = acciones USA (S&P 500)      | TLT = bonos del Tesoro USA largos
#   GLD  = oro                          | EEM = acciones emergentes
#   MSFT = una acción individual (riesgo idiosincrático, para ver cómo
#          la diversificación lo reduce)
TICKERS = ["SPY", "TLT", "GLD", "EEM", "MSFT"]

FECHA_INICIO = "2021-07-01"   # ventana de 5 años (máximo que permite la consigna:
FECHA_FIN = "2026-07-01"      # más observaciones = estimación más estable de la
                              # matriz de covarianzas, que tiene 15 parámetros)

N_CARTERAS = 10_000           # cantidad de carteras aleatorias a simular
TASA_LIBRE_RIESGO = 0.04      # 4% anual (aprox. T-bill USA). Necesaria para el
                              # Sharpe: mide el retorno EN EXCESO de lo que paga
                              # un activo sin riesgo, por unidad de volatilidad.
DIAS_HABILES = 252            # días de mercado por año, para anualizar
CSV_CACHE = "datos_precios.csv"
SEMILLA = 42                  # para que la simulación sea reproducible

# ----------------------------------------------------------------------------
# PASO 1: OBTENER PRECIOS (descarga o cache local)
# ----------------------------------------------------------------------------
# La primera corrida descarga de Yahoo Finance y guarda un CSV. Las siguientes
# leen el CSV: el trabajo queda reproducible aunque Yahoo cambie o no haya
# internet, y quien clone el repo obtiene exactamente los mismos números.
if os.path.exists(CSV_CACHE):
    print(f"Leyendo precios desde cache local ({CSV_CACHE})...")
    precios = pd.read_csv(CSV_CACHE, index_col=0, parse_dates=True)
else:
    print("Descargando precios de Yahoo Finance...")
    import yfinance as yf
    # auto_adjust=True -> precios ajustados por dividendos y splits.
    # Sin esto, un dividendo parecería una caída del precio y el retorno
    # calculado estaría distorsionado.
    precios = yf.download(TICKERS, start=FECHA_INICIO, end=FECHA_FIN,
                          auto_adjust=True)["Close"]
    precios = precios.dropna()      # descartar días donde falta algún activo
    if precios.empty:
        raise SystemExit("ERROR: la descarga vino vacía (¿sin internet o "
                         "Yahoo caído?). Reintentá más tarde.")
    precios.to_csv(CSV_CACHE)
    print(f"Precios guardados en {CSV_CACHE}")

precios = precios[TICKERS]          # fijar el orden de las columnas
print(f"{len(precios)} días de datos, de {precios.index[0].date()} "
      f"a {precios.index[-1].date()}\n")

# ----------------------------------------------------------------------------
# PASO 2: RETORNOS DIARIOS
# ----------------------------------------------------------------------------
# Se trabaja con retornos (variaciones %) y no con precios: los precios de
# distintos activos no son comparables entre sí ni estadísticamente estables;
# los retornos sí, y son el insumo que asume el modelo de Markowitz.
retornos = precios.pct_change().dropna()

# ----------------------------------------------------------------------------
# PASO 3: ESTIMAR INSUMOS DEL MODELO (anualizados)
# ----------------------------------------------------------------------------
# ANUALIZACIÓN: los datos son diarios, pero la convención financiera es
# expresar retorno y riesgo en términos anuales (comparable con la tasa libre
# de riesgo, que es anual, y con cualquier informe de mercado).
#   - retorno esperado: media diaria x 252 (los retornos se acumulan ~aditivamente)
#   - varianza/covarianza: x 252 (las varianzas de retornos independientes se
#     suman). La VOLATILIDAD anual queda como sqrt(varianza anual), es decir
#     escala con sqrt(252), no con 252.
mu = retornos.mean() * DIAS_HABILES        # vector de retornos esperados anuales
cov = retornos.cov() * DIAS_HABILES        # matriz var-cov anual (5x5)
correl = retornos.corr()                   # matriz de correlación (para graficar)

print("Retorno esperado anual por activo (%):")
print((mu * 100).round(2), "\n")
print("Volatilidad anual por activo (%):")
print(pd.Series(np.sqrt(np.diag(cov)) * 100, index=TICKERS).round(2), "\n")

# ----------------------------------------------------------------------------
# PASO 4: SIMULAR CARTERAS ALEATORIAS (Monte Carlo)
# ----------------------------------------------------------------------------
# Cada cartera es un vector de 5 pesos >= 0 que suman 1 (sin venta en corto).
# La distribución Dirichlet genera pesos que cubren el espacio de forma pareja.
rng = np.random.default_rng(SEMILLA)
pesos = rng.dirichlet(np.ones(len(TICKERS)), N_CARTERAS)   # (10000, 5)

# Retorno de cada cartera: promedio ponderado de los retornos esperados.
rets_carteras = pesos @ mu.values

# Volatilidad de cada cartera: sqrt(w' Σ w). Acá actúa la covarianza:
# combinar activos poco correlacionados reduce el riesgo total por debajo
# del promedio de los riesgos individuales -> beneficio de diversificar.
vols_carteras = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov.values, pesos))

# Ratio de Sharpe: exceso de retorno sobre la tasa libre de riesgo,
# por unidad de riesgo asumido.
sharpe_carteras = (rets_carteras - TASA_LIBRE_RIESGO) / vols_carteras

# ----------------------------------------------------------------------------
# PASO 5: IDENTIFICAR LOS DOS PORTFOLIOS ÓPTIMOS
# ----------------------------------------------------------------------------
i_sharpe = sharpe_carteras.argmax()   # máximo Sharpe (tangency portfolio)
i_minvol = vols_carteras.argmin()     # mínima varianza

def describir(nombre, i):
    print(f"--- {nombre} ---")
    print(f"  Retorno esperado: {rets_carteras[i]*100:6.2f} % anual")
    print(f"  Volatilidad:      {vols_carteras[i]*100:6.2f} % anual")
    print(f"  Sharpe:           {sharpe_carteras[i]:6.3f}")
    print("  Ponderaciones:")
    for ticker, w in zip(TICKERS, pesos[i]):
        print(f"    {ticker:5s} {w*100:5.1f} %")
    print()

describir("PORTFOLIO DE MÁXIMO SHARPE", i_sharpe)
describir("PORTFOLIO DE MÍNIMA VARIANZA", i_minvol)

# ----------------------------------------------------------------------------
# PASO 6: GRÁFICOS
# ----------------------------------------------------------------------------
# (a) Frontera eficiente: nube de carteras simuladas en el plano riesgo-retorno.
#     El borde superior-izquierdo de la nube ES la frontera eficiente; los
#     puntos interiores son carteras dominadas (existe otra con más retorno
#     al mismo riesgo).
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(vols_carteras * 100, rets_carteras * 100,
                c=sharpe_carteras, cmap="viridis", s=4, alpha=0.6)
fig.colorbar(sc, label="Ratio de Sharpe")
ax.scatter(vols_carteras[i_sharpe] * 100, rets_carteras[i_sharpe] * 100,
           c="red", marker="*", s=400, edgecolors="black", zorder=3,
           label=f"Máx. Sharpe ({sharpe_carteras[i_sharpe]:.2f})")
ax.scatter(vols_carteras[i_minvol] * 100, rets_carteras[i_minvol] * 100,
           c="blue", marker="*", s=400, edgecolors="black", zorder=3,
           label=f"Mín. varianza (vol {vols_carteras[i_minvol]*100:.1f}%)")
# Activos individuales, para ver que las carteras diversificadas los dominan
for t in TICKERS:
    ax.scatter(np.sqrt(cov.loc[t, t]) * 100, mu[t] * 100,
               marker="X", s=120, c="dimgray", zorder=3)
    ax.annotate(t, (np.sqrt(cov.loc[t, t]) * 100, mu[t] * 100),
                xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.set_xlabel("Volatilidad anual (%)")
ax.set_ylabel("Retorno esperado anual (%)")
ax.set_title(f"Frontera eficiente — {N_CARTERAS:,} carteras simuladas "
             f"({precios.index[0].year}-{precios.index[-1].year})")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("frontera_eficiente.png", dpi=150)
print("Guardado frontera_eficiente.png")

# (b) Comparación de ponderaciones (punto 3 de la consigna)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(TICKERS))
ancho = 0.35
ax.bar(x - ancho/2, pesos[i_sharpe] * 100, ancho, label="Máx. Sharpe",
       color="firebrick")
ax.bar(x + ancho/2, pesos[i_minvol] * 100, ancho, label="Mín. varianza",
       color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels(TICKERS)
ax.set_ylabel("Ponderación (%)")
ax.set_title("Ponderaciones: máximo Sharpe vs. mínima varianza")
ax.legend()
fig.tight_layout()
fig.savefig("ponderaciones.png", dpi=150)
print("Guardado ponderaciones.png")

# (c) Matriz de correlación: muestra POR QUÉ diversificar funciona
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(correl, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(TICKERS)))
ax.set_xticklabels(TICKERS)
ax.set_yticks(range(len(TICKERS)))
ax.set_yticklabels(TICKERS)
for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        ax.text(j, i, f"{correl.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=9)
fig.colorbar(im, label="Correlación")
ax.set_title("Matriz de correlación (retornos diarios)")
fig.tight_layout()
fig.savefig("correlaciones.png", dpi=150)
print("Guardado correlaciones.png")

print("\nListo. Abrí los .png para ver los resultados.")
