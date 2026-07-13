# 🏈 NFL Probability Engine

> Probabilistic forecasting of NFL regular season and playoffs using calibrated ELO ratings,
> logistic regression, and Monte Carlo simulation.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Fase%205%20pendiente-yellow)

---

## ¿Qué hace este proyecto?

Este proyecto construye un **motor de predicción probabilística** para la NFL que:

1. Calcula ratings **ELO** para los 32 equipos usando 26 temporadas de historia (1999–2025)
2. Entrena y compara un modelo de **regresión logística** y **Gradient Boosting** contra el ELO puro
3. Simula la **temporada 2026 completa** (regular season + playoffs) vía Monte Carlo (10,000 simulaciones)
4. Produce distribuciones de wins, probabilidades de playoffs, campeonatos de división y de Super Bowl
5. Evalúa el desempeño del modelo contra las **líneas de Vegas** como benchmark externo
6. Genera visualizaciones de estilo Apple dark-mode y un carrusel listo para LinkedIn

---

## Arquitectura del sistema

```
Datos históricos 1999–2025 (nflreadpy)
         │
         ▼
┌─────────────────────┐
│   Sistema ELO       │  ← rating por equipo, partido a partido
│   (núcleo)          │     REG + WC + DIV + CON + SB
└────────┬────────────┘
         │  elo_diff = R_home - R_away + HOME_ADVANTAGE
         ▼
┌─────────────────────┐
│ Regresión logística │  ← elo_diff + rest + div_game + week_norm
│  vs. Gradient Boost. │     (comparación, no reemplaza al ELO)
└────────┬────────────┘
         │  ELO puro gana en Brier Score → motor final usa ELO
         ▼
┌─────────────────────┐
│  Motor Monte Carlo  │  ← 10,000 simulaciones de temporada 2026
│  (simulación)       │     regular season + bracket de playoffs
└────────┬────────────┘
         │
         ▼
   Distribución de wins · % playoffs · % División · % Super Bowl
         │
         ▼
┌─────────────────────┐
│   Visualizaciones    │  ← standings, bracket, calendario por equipo,
│   + carrusel LinkedIn│     carrusel de 10 slides
└─────────────────────┘
```

---

## Resultados actuales

### Brier Score — evaluado sobre 4,175 juegos REG (2010–2025)

| Modelo | Brier Score | Accuracy | Gap vs. nulo cerrado |
|---|---|---|---|
| Modelo nulo (50/50) | 0.2500 | ~56% | 0% |
| ELO con parámetros FiveThirtyEight | 0.2267 | — | 59.5% |
| **ELO calibrado (propio)** | **0.2235** | **64.0%** | **67.9%** |
| Regresión logística (todos los features) | 0.2248 | 64.4% | — |
| Gradient Boosting | 0.2250 | 62.9% | — |
| Vegas (benchmark externo) | 0.2114 | ~67% | 100% |

**Hallazgo clave (Notebook 03):** el ELO puro supera tanto a la regresión logística como
al Gradient Boosting. Los features adicionales (`home_rest`, `away_rest`, `div_game`,
`week_norm`) tienen coeficientes β 12–44× menores que `elo_diff`; un modelo con solo
`elo_diff` (BS=0.2246) prácticamente iguala al modelo completo (BS=0.2248). Tuning de
regularización (C entre 0.001 y 10.0) no mejora el resultado.

**Interpretación:** el ELO captura casi toda la señal predictiva disponible en variables
estructurales públicas. La varianza restante frente a Vegas requiere información que no
tenemos (lesiones en tiempo real, rotaciones de QB).

**Decisión de diseño:** el motor Monte Carlo usa **ELO puro**, no el modelo logístico.

---

## Parámetros del sistema ELO — todos calibrados desde datos

```python
ELO_INICIAL       = 1500    # convención estándar
HOME_ADVANTAGE    = 42.2    # derivado: -400 × log10((1/0.563) - 1)
K_REGULAR         = 40      # grid search · validado out-of-sample 2020–2025
K_PLAYOFFS        = 48      # K × 1.2
REGRESSION_FACTOR = 0.40    # grid search · validado out-of-sample 2020–2025
GLOBAL_MEAN       = 1503.1  # media empírica calculada
HOME_ADVANTAGE_SB = 0       # Super Bowl en cancha neutral
```

**Proceso de calibración:**
- `HOME_ADVANTAGE`: despeje algebraico desde el home win rate observado
- `GLOBAL_MEAN`: media empírica corriendo el sistema con global_mean=1500
- `K` y `REGRESSION_FACTOR`: grid search simultáneo (28 combinaciones), calibración
  2010–2019 → validación 2020–2025

### Comparación vs. FiveThirtyEight

| Parámetro | FiveThirtyEight | Nuestro sistema | Motivo de diferencia |
|---|---|---|---|
| `HOME_ADVANTAGE` | 65 | **42.2** | Era 1990–2015 vs. 1999–2025 |
| `K_REGULAR` | 20 | **40** | NFL moderna más volátil (salary cap, libre agencia) |
| `K_PLAYOFFS` | 24 | **48** | Proporcional a K_REGULAR |
| `REGRESSION_FACTOR` | 0.33 | **0.40** | Mayor paridad, equipos cambian más rápido |
| `GLOBAL_MEAN` | 1505 | **1503.1** | Media empírica real del sistema |

Diferencia de Brier Score vs. parámetros FiveThirtyEight: mejora de 0.0032 out-of-sample.

---

## Decisiones de diseño clave

| Decisión | Elección | Alternativa descartada | Razón |
|---|---|---|---|
| ELO scope | REG + WC + DIV + CON + SB | Solo REG | Maximiza info entre temporadas |
| HOME_ADVANTAGE Super Bowl | 0 | 42.2 | Cancha neutral |
| Vegas en el modelo | Solo benchmark | Feature del modelo | Razonamiento circular |
| `qb_name` | No es feature | Incluir como dummy | El ELO lo captura indirectamente |
| 2020 COVID | `sample_weight = 0.3` | Excluir | Mantener N, reducir influencia |
| Validación | Walk-forward expandible | K-Fold estándar | Evita leakage temporal |
| `StandardScaler` | Fiteado solo en train por fold | Global | Evita leakage de escala |
| Motor de predicción 2026 | ELO puro | Regresión logística | El ELO supera a LR y GB en Brier Score |
| `K_PLAYOFFS` | K × 1.2 | Calibración independiente | Pocos juegos de playoffs (~338) para estimar K aparte |

### Modelo predictivo — features

| Feature | Incluido | Motivo |
|---|---|---|
| `elo_diff` | ✅ | Feature principal · captura la gran mayoría del poder predictivo |
| `home_rest` | ✅ | Días de descanso del local |
| `away_rest` | ✅ | Días de descanso del visitante |
| `div_game` | ✅ | Rivalidades divisionales tienen dinámica propia |
| `week_norm` | ✅ | `week / max_week` · normaliza transición de 16 a 17 juegos (2021) |
| `home_moneyline` | ❌ | Solo benchmark externo · incluirlo sería razonamiento circular |
| `qb_name` | ❌ | El ELO lo captura indirectamente vía resultados |
| `temp` / `wind` | ❌ | Alto % de missing · señal débil vs. costo de imputación |

---

## Hallazgos del EDA

- **Dataset:** 7,276 partidos × 46 columnas · temporadas 1999–2025
- **Home win rate:** 56.3% global · tendencia declinante era post-COVID (2021–2025: ~53–54%)
- **2020 COVID:** outlier estructural (home win rate ~49%) · `sample_weight = 0.3` en training
- **Transición 2021:** temporada de 17 juegos → `week_norm = week / max_week`
- **Vegas missing:** concentrado en 1999–2005 · benchmark aplica 2007–2025
- **Franquicias unificadas:** OAK→LV, SD→LAC, STL→LA (normalizadas por `nflreadpy`)

---

## Predicciones 2026 — highlights

```
Top favoritos al Super Bowl:
  SEA: 12.8%  ← campeón defensor Super Bowl LX
  DEN:  7.8%
  NE:   7.4%
  HOU:  6.8%

Equipos que más cayeron vs. expectativa histórica:
  KC: 34.7% playoffs · 1.6% Super Bowl  ← era de dominancia en declive

División más competitiva:
  NFC West: SEA (80.9%) + LA (60.7%) + SF (56.0%) — tres equipos >56% playoffs

Wild Card más probable:
  AFC: HOU (bye) · DEN vs LAC · NE vs JAX · BAL vs BUF
  NFC: SEA (bye) · PHI vs MIN · DET vs SF · ATL vs LA
```

---

## Estructura del repositorio

```
nfl-probability-engine/
├── Notebooks/
│   ├── 01_exploratory_analysis.ipynb   ✅ EDA · calibración de parámetros
│   ├── 02_elo_system.ipynb             ✅ Sistema ELO · validación
│   ├── 03_logistic_model.ipynb         ✅ Regresión logística vs. GB
│   └── 04_monte_carlo.ipynb            ✅ Simulación temporada 2026
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       — load_nfl_data()
│   ├── elo.py               — sistema ELO completo (cálculo, calibración, backtesting)
│   ├── model.py             — build_features, walk_forward_validation, train_final_model
│   ├── simulation.py        — run_monte_carlo, simulate_regular_season, bracket de playoffs
│   └── visualization.py     — standings, bracket, calendario por equipo, carrusel LinkedIn
│
├── data/
│   └── processed/
│       ├── elo_history.csv            — ELO partido a partido 1999–2025
│       ├── elo_final_2025.csv         — ratings al cierre de 2025
│       ├── model_features.csv         — dataset con features construidos
│       ├── model_coefficients.csv     — coeficientes β finales
│       ├── predictions_2026.csv       — resultados Monte Carlo agregados
│       ├── wins_distribution_2026.csv — distribución de wins por simulación
│       └── game_results_2026.csv      — probabilidad de victoria por partido
│
├── results/
│   ├── regular_season_standings_2026.png
│   ├── wildcard_bracket_2026.png
│   └── carousel/                      — slides LinkedIn (slide_01.png … slide_10.png)
│
├── streamlit_app.py         ⏳ Pendiente (Fase 5)
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Fases del proyecto

| Fase | Notebook | Estado | Output clave |
|---|---|---|---|
| 1 · EDA | `01_exploratory_analysis.ipynb` | ✅ Completo | Parámetros calibrados · Decision Log |
| 2 · ELO | `02_elo_system.ipynb` | ✅ Completo | `elo_history.csv` · `elo_final_2025.csv` · BS=0.2235 |
| 3 · Modelo | `03_logistic_model.ipynb` | ✅ Completo | Coeficientes β · BS=0.2248 · comparación LR vs. GB |
| 4 · Simulación | `04_monte_carlo.ipynb` | ✅ Completo | `predictions_2026.csv` · visualizaciones |
| 5 · Dashboard | `streamlit_app.py` | ⏳ Pendiente | App interactiva |

---

## Funciones implementadas

### `src/data_loader.py`
```python
load_nfl_data(seasons=range(1999, 2026))
# → DataFrame limpio con franquicias normalizadas y home_team_win
```

### `src/elo.py`
```python
estimate_home_advantage(df, last_n_seasons=None, game_types=['REG'])
expected_win_prob(elo_home, elo_away, home_advantage)
update_elo(elo_home, elo_away, result, k, home_advantage)
apply_season_regression(elo_ratings, regression_factor, global_mean)
run_elo_system(df, home_advantage, regression_factor, global_mean, k_regular, k_playoffs)
backtest_home_advantage(df, ha_candidates, test_seasons)
calibrate_global_mean(df, home_advantage, regression_factor)
backtest_regression_factor(df, alphas, test_seasons, home_advantage, global_mean)
grid_search_elo(df, k_values, alphas, test_seasons, home_advantage, global_mean)
```

### `src/model.py`
```python
FEATURES = ['elo_diff', 'home_rest', 'away_rest', 'div_game', 'week_norm']

build_features(history_df, schedules_df, home_advantage)
walk_forward_validation(df, test_seasons, feature_cols, model=None, scale_features=None)
# → acepta cualquier modelo sklearn; infiere si necesita StandardScaler
train_final_model(df, feature_cols)
```

### `src/simulation.py`
```python
# Constantes de liga
CONFERENCES, DIVISIONS, TEAM_CONF, TEAM_DIV

run_monte_carlo(schedule_df, final_elos_2025, home_advantage, k_regular,
                k_playoffs, regression_factor, global_mean,
                n_sims=10_000, seed=42)
# → results_df, wins_dist, game_results_df

simulate_regular_season(schedule_df, elos, home_advantage, k_regular, rng)
# → elos, records, game_results  (lista de 0/1 por partido)

determine_playoff_seeds(records, rng)
# → {'AFC': [s1..s7], 'NFC': [s1..s7]}

simulate_playoff_bracket(seeds, elos, home_advantage, k_playoffs, rng)
# → sb_winner, conf_champs, elos
```

### `src/visualization.py`
```python
# Helpers internos
_get_logo(team, size=(45,45))     # descarga, redimensiona y cachea logos ESPN
_add_logo(ax, team, x, y, zoom)   # coloca logo en eje matplotlib

# Funciones públicas
plot_regular_season_standings(results_df, bg_color='#1c1c1e', save_path=None)
plot_wildcard_bracket(results_df, bg_color='#1c1c1e', save_path=None)
plot_team_schedule(team, game_results_df, bg_color='#1c1c1e', save_path=None)
generate_linkedin_carousel(results_df, save_dir='../results/carousel', bg_color='#1c1c1e')
```

---

## Design system para visualizaciones

```python
# Paleta consistente en todo el proyecto
BG_COLOR   = '#1c1c1e'    # Apple dark gray
CARD_BG    = '#2c2c2e'    # Apple secondary
AFC_COLOR  = '#013369'    # NFL azul oficial AFC
NFC_COLOR  = '#D50A0A'    # NFL rojo oficial NFC
GOLD_COLOR = '#f5c518'    # Acento dorado (líderes de división)

# Logos: descargados desde ESPN via nflreadpy.load_teams()
# Resize a 45×45px antes de OffsetImage con zoom=0.50
# ABBR_MAP = {'LA': 'LAR', 'JAX': 'JAC'}  ← corrección de abreviaciones
```

---

## Métricas de evaluación

```
Métrica principal    : Brier Score  (calibración probabilística)
Métricas secundarias : Log-Loss · Accuracy · Calibration Curve
Benchmark            : Vegas moneylines sin vig (2010–2025)
Validación           : Ventana expandible (walk-forward)
    → Train 2010–2019 | Test 2020–2025 (calibración ELO)
```

---

## Stack tecnológico

```
Python 3.11+
├── nflreadpy        — fuente de datos (reemplaza a nfl_data_py, deprecado)
├── pandas / numpy    — manipulación de datos
├── scikit-learn      — regresión logística · Gradient Boosting · métricas
├── scipy             — pruebas estadísticas
├── matplotlib        — visualizaciones
└── streamlit         — dashboard interactivo (Fase 5)
```

---

## Instalación

```bash
git clone https://github.com/sebastianBP26/nfl-probability-engine.git
cd nfl-probability-engine
pip install -r requirements.txt
```

Ejecutar los notebooks en orden:

```
01_exploratory_analysis.ipynb  →  02_elo_system.ipynb  →  03_logistic_model.ipynb  →  04_monte_carlo.ipynb
```

---

## Próximos pasos

- [ ] Explorar actualización in-season del ELO conforme avance la temporada 2026

---

## Referencias

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
  → [Internet Archive](https://archive.org/details/ratingofchesspla00unse)
- FiveThirtyEight NFL ELO Data & Code.
  → https://github.com/fivethirtyeight/data/tree/master/nfl-elo

---

*Proyecto de portafolio · Sebastián Barroso · 2026*
