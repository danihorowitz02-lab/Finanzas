# Frontera Eficiente de Markowitz

Construcción de la frontera eficiente con datos reales de mercado, e identificación del portfolio de máximo Sharpe y el de mínima varianza mediante simulación Monte Carlo.

## Qué hace el script

`frontera_eficiente.py` ejecuta el análisis completo en 6 pasos:

1. **Datos**: descarga 5 años de precios ajustados (jul 2021 – jul 2026) de Yahoo Finance vía `yfinance` y los guarda en `datos_precios.csv`. Las corridas siguientes leen el CSV, lo que hace el análisis reproducible sin depender de internet.
2. **Retornos**: calcula retornos diarios porcentuales a partir de los precios.
3. **Estimación**: retorno esperado (media diaria × 252) y matriz de varianzas-covarianzas (covarianza diaria × 252), ambos anualizados para expresarlos en unidades estándar comparables con la tasa libre de riesgo.
4. **Simulación**: genera 10.000 carteras con pesos aleatorios (distribución Dirichlet, sin venta en corto) y calcula retorno, volatilidad y ratio de Sharpe de cada una.
5. **Optimización**: identifica el portfolio de máximo Sharpe (tangency portfolio) y el de mínima varianza, e imprime sus ponderaciones por consola.
6. **Gráficos**: guarda tres PNG — la frontera eficiente, la comparación de ponderaciones y la matriz de correlación.

## Activos

| Ticker | Clase |
|--------|-------|
| SPY | Acciones USA (S&P 500) |
| TLT | Bonos del Tesoro USA de larga duración |
| GLD | Oro |
| EEM | Acciones de mercados emergentes |
| MSFT | Acción individual (riesgo idiosincrático) |

Elegidos buscando baja correlación entre clases de activos: el beneficio de la diversificación —el corazón del modelo de Markowitz— solo aparece cuando las correlaciones son menores a 1.

## Uso

```bash
pip install -r requirements.txt
python frontera_eficiente.py
```

Salida: `frontera_eficiente.png`, `ponderaciones.png`, `correlaciones.png` y los resultados numéricos por consola.

## Supuestos

- Tasa libre de riesgo: 4% anual (aprox. T-bill USA).
- 252 días hábiles por año para anualizar.
- Sin venta en corto (pesos ≥ 0).
- μ y Σ estimados sobre la muestra histórica se asumen válidos hacia adelante — el supuesto más frágil del modelo en la práctica.
