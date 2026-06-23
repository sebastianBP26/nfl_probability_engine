# 🏈 NFL Probability Engine

> Probabilistic forecasting of NFL regular season games using ELO ratings,
> logistic regression, and Monte Carlo simulation.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-En%20desarrollo-yellow)

---

## ¿Qué hace este proyecto?

Este proyecto construye un **motor de predicción probabilística** para la NFL que:

1. Calcula ratings **ELO** para los 32 equipos usando 26 temporadas de historia (1999–2025)
2. Entrena un modelo de **regresión logística** para predecir resultados de temporada regular
3. Simula la **temporada 2026 completa** via Monte Carlo (10,000 simulaciones)
4. Produce distribuciones de victorias, probabilidades de playoffs y campeonatos de división
5. Evalúa el desempeño del modelo contra las **líneas de Vegas** como benchmark externo

---

## Arquitectura del sistema

```
Datos históricos 1999–2025
         │
         ▼
┌─────────────────────┐
│   Sistema ELO       │  ← rating por equipo, partido a partido
│   (núcleo)          │
└────────┬────────────┘
         │  elo_diff = R_home - R_away + HOME_ADVANTAGE
         ▼
┌─────────────────────┐
│ Regresión logística │  ← elo_diff + rest + div_game + week_norm
│   (modelo)          │
└────────┬────────────┘
         │  P(home wins) por partido
         ▼
┌─────────────────────┐
│  Motor Monte Carlo  │  ← 10,000 simulaciones de temporada 2026
│  (simulación)       │
└────────┬────────────┘
         │
         ▼
   Distribución de wins · % playoffs · % División
```

---

## Decisiones de diseño clave

### Sistema ELO

| Parámetro | Valor | Justificación |
|---|---|---|
| `ELO_INICIAL` | 1500 | Convención estándar |
| `HOME_ADVANTAGE` | 42.2 pts | Calibrado desde home win rate observado 1999–2025 (56.3%) |
| `HOME_ADVANTAGE` Super Bowl | 0 | Cancha neutral |
| `K_REGULAR` | 20 | Validado por FiveThirtyEight (2014) · confirmado con backtesting |
| `K_PLAYOFFS` | 24 | K × 1.2 · mayor señal informativa por partido eliminatorio |
| Regresión entre temporadas | 33% | `ELO_t+1 = 0.67 × ELO_final + 0.33 × 1505` |
| Scope de actualización | REG + WC + DIV + CON + SB | Maximiza información entre temporadas |

> **Nota:** `HOME_ADVANTAGE = 65` (FiveThirtyEight) fue calibrado en era 1990–2015.
> Los datos 1999–2025 muestran una tendencia declinante del home advantage,
> por lo que se recalibra desde los datos propios.

### Modelo predictivo

| Feature | Incluido | Motivo |
|---|---|---|
| `elo_diff` | ✅ | Feature principal · captura ~85–90% del poder predictivo |
| `home_rest` | ✅ | Días de descanso del local |
| `away_rest` | ✅ | Días de descanso del visitante |
| `div_game` | ✅ | Rivalidades divisionales tienen dinámica propia |
| `week_norm` | ✅ | `week / max_week` · normaliza transición de 16 a 17 juegos (2021) |
| `home_moneyline` | ❌ | Solo benchmark externo · incluirlo sería razonamiento circular |
| `qb_name` | ❌ | El ELO lo captura indirectamente vía resultados |
| `temp` / `wind` | ❌ | 28% missing · señal débil vs. costo de imputación |

### Benchmark de referencia

Las líneas de Vegas (moneylines convertidas a probabilidades, sin vig) sirven
como benchmark externo de Brier Score. **No son features del modelo.**

```
Benchmark válido: 2007–2025 (~4,500 juegos)
Missing concentrado en: 1999–2005
```

---

## Estructura del repositorio

```
nfl-probability-engine/
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # EDA · calibración de parámetros
│   ├── 02_elo_system.ipynb             # Implementación y validación del ELO
│   ├── 03_model.ipynb                  # Regresión logística · backtesting
│   └── 04_monte_carlo.ipynb            # Simulación temporada 2026
│
├── src/
│   ├── data_loader.py                  # Carga y limpieza de datos
│   ├── elo.py                          # Sistema ELO (cálculo y actualización)
│   ├── model.py                        # Regresión logística y evaluación
│   ├── simulation.py                   # Motor Monte Carlo
│   └── utils.py                        # Funciones auxiliares
│
├── data/
│   ├── raw/                            # No versionado (se regenera con nfl_data_py)
│   └── processed/
│       ├── elo_history.csv             # ELO partido a partido 1999–2025
│       └── elo_final_2025.csv          # Rating final por equipo al cierre de 2025
│
├── results/                            # Outputs de predicciones 2026
├── README.md
└── requirements.txt
```

---

## Fases del proyecto

| Fase | Notebook | Estado | Output |
|---|---|---|---|
| 1 · EDA | `01_exploratory_analysis` | ✅ Completo | Parámetros calibrados · Decision Log |
| 2 · ELO | `02_elo_system` | 🔄 En progreso | `elo_history.csv` · `elo_final_2025.csv` |
| 3 · Modelo | `03_model` | ⏳ Pendiente | Coeficientes β · Brier Score vs. Vegas |
| 4 · Simulación | `04_monte_carlo` | ⏳ Pendiente | Distribución de wins temporada 2026 |
| 5 · Dashboard | `streamlit_app` | ⏳ Pendiente | App interactiva |

---

## Hallazgos del EDA

- **Dataset:** 7,276 partidos × 46 columnas · temporadas 1999–2025
- **Home win rate:** 56.3% global · tendencia declinante post-2020 (~53–54%)
- **2020 COVID:** outlier estructural (home win rate ~49%) · down-weighted en training
- **Transición 2021:** temporada de 17 juegos (256 → 272 partidos/temporada)
- **Franquicias unificadas:** OAK→LV (2020), SD→LAC (2017), STL→LA (2016)

### Home win rate por tipo de partido

| game_type | Home Win Rate | Observación |
|---|---|---|
| REG | 56.1% | Base del modelo |
| WC | 58.6% | Sesgo de selección por seed |
| DIV | 69.4% | Sesgo mayor |
| CON | 66.7% | Sesgo mayor |
| SB | 37.0% | Cancha neutral · ruido sobre 26 observaciones |

---

## Métricas de evaluación

```
Métrica principal : Brier Score  (calibración probabilística)
Métricas secundarias : Log-Loss · Accuracy · Calibration Curve
Benchmark : Vegas moneylines (Brier Score de referencia)
Validación : Ventana expandible
    → Train 1999–2021 | Test 2022–2024
    → Train 1999–2022 | Test 2023–2024
```

---

## Stack tecnológico

```
Python 3.11+
├── nfl_data_py      — fuente de datos
├── pandas / numpy   — manipulación de datos
├── scikit-learn     — regresión logística · métricas
├── scipy            — pruebas estadísticas
├── matplotlib / plotly — visualizaciones
└── streamlit        — dashboard interactivo (Fase 5)
```

---

## Instalación

```bash
git clone https://github.com/sebastianBP26/nfl-probability-engine.git
cd nfl-probability-engine
pip install -r requirements.txt
```

---

## Referencias

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
  → [Internet Archive](https://archive.org/details/ratingofchesspla00unse)
- Silver, N. et al. (2014). *How Our NFL Predictions Work*. FiveThirtyEight.
  → https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
- FiveThirtyEight NFL ELO Data & Code.
  → https://github.com/fivethirtyeight/data/tree/master/nfl-elo

---

*Proyecto de portafolio · Sebastián Barroso · 2026*
