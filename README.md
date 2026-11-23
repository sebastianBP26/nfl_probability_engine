# 🏈 NFL Probability Engine

> Probabilistic forecasting of NFL regular season and playoffs using calibrated ELO ratings,
> logistic regression, and Monte Carlo simulation.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%205%20pending-yellow)

---

## What does this project do?

This project builds a **probabilistic forecasting engine** for the NFL that:

1. Computes **ELO** ratings for all 32 teams using 26 seasons of history (1999–2025)
2. Trains and benchmarks a **logistic regression** model and **Gradient Boosting** against pure ELO
3. Simulates the **full 2026 season** (regular season + playoffs) via Monte Carlo (10,000 simulations)
4. Produces win distributions, playoff odds, division titles, and Super Bowl probabilities
5. Evaluates model performance against **Vegas lines** as an external benchmark
6. Generates Apple dark-mode style visualizations and a LinkedIn-ready carousel
7. Updates ELO **in-season**, week by week, with real results and publishes weekly picks,
   model performance and Instagram-ready images (series *"Pronosticando la NFL con un modelo de ELO"*)

---

## System architecture

```
Historical data 1999–2025 (nflreadpy)
         │
         ▼
┌─────────────────────┐
│    ELO system       │  ← rating per team, game by game
│      (core)         │     REG + WC + DIV + CON + SB
└────────┬────────────┘
         │  elo_diff = R_home - R_away + HOME_ADVANTAGE
         ▼
┌─────────────────────┐
│ Logistic regression  │  ← elo_diff + rest + div_game + week_norm
│    vs. Grad. Boost.  │     (benchmark comparison, not a replacement)
└────────┬────────────┘
         │  Pure ELO wins on Brier Score → final engine uses ELO
         ▼
┌─────────────────────┐
│  Monte Carlo engine  │  ← 10,000 simulations of the 2026 season
│    (simulation)      │     regular season + playoff bracket
└────────┬────────────┘
         │
         ▼
   Win distribution · playoff % · division % · Super Bowl %
         │
         ▼
┌─────────────────────┐
│   Visualizations      │  ← standings, bracket, per-team schedule,
│   + LinkedIn carousel │     10-slide carousel
└─────────────────────┘
```

### In-season weekly loop (2026)

```
Real 2026 results (nflreadpy · every Tuesday after MNF)
         │
         ▼
┌─────────────────────┐
│  Weekly ELO update   │  ← week 1: end-of-2025 ELO + season regression
│ (weekly_probabilities)│     week N: ELO updated with real results of weeks 1..N-1
└────────┬────────────┘
         │  win probability per game with the ELO in force at that date
         ▼
┌─────────────────────┐
│  Evaluation          │  ← record, Brier, log loss, calibration,
│  (weekly_content)    │     comparison vs. Vegas favorite
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Weekly content      │  ← picks, ELO movers, top 5, highlighted games,
│  reels/semana_XX/    │     one story per game + resumen.md script
└─────────────────────┘
```

---

## Current results

### Brier Score — evaluated on 4,175 REG games (2010–2025)

| Model | Brier Score | Accuracy | Gap vs. null closed |
|---|---|---|---|
| Null model (50/50) | 0.2500 | ~56% | 0% |
| ELO with FiveThirtyEight parameters | 0.2267 | — | 59.5% |
| **Calibrated ELO (own)** | **0.2235** | **64.0%** | **67.9%** |
| Logistic regression (all features) | 0.2248 | 64.4% | — |
| Gradient Boosting | 0.2250 | 62.9% | — |
| Vegas (external benchmark) | 0.2114 | ~67% | 100% |

**Key finding (Notebook 03):** pure ELO outperforms both logistic regression and Gradient
Boosting. The additional features (`home_rest`, `away_rest`, `div_game`, `week_norm`) have
β coefficients 12–44x smaller than `elo_diff`; a model with only `elo_diff` (BS=0.2246)
essentially matches the full model (BS=0.2248). Regularization tuning (C between 0.001 and
10.0) doesn't improve results.

**Interpretation:** ELO captures nearly all the predictive signal available in public
structural variables. The remaining variance versus Vegas requires information we don't
have (real-time injuries, QB rotations).

**Design decision:** the Monte Carlo engine uses **pure ELO**, not the logistic model.

---

## ELO system parameters — all calibrated from data

```python
INITIAL_ELO       = 1500    # standard convention
HOME_ADVANTAGE    = 42.2    # derived: -400 × log10((1/0.563) - 1)
K_REGULAR         = 40      # grid search · validated out-of-sample 2020–2025
K_PLAYOFFS        = 48      # K × 1.2
REGRESSION_FACTOR = 0.40    # grid search · validated out-of-sample 2020–2025
GLOBAL_MEAN       = 1503.1  # empirically computed mean
HOME_ADVANTAGE_SB = 0       # Super Bowl on neutral field
```

**Calibration process:**
- `HOME_ADVANTAGE`: algebraic derivation from observed home win rate
- `GLOBAL_MEAN`: empirical mean from running the system with global_mean=1500
- `K` and `REGRESSION_FACTOR`: simultaneous grid search (28 combinations), calibration
  on 2010–2019 → validation on 2020–2025

### Comparison vs. FiveThirtyEight

| Parameter | FiveThirtyEight | Our system | Reason for difference |
|---|---|---|---|
| `HOME_ADVANTAGE` | 65 | **42.2** | 1990–2015 era vs. 1999–2025 |
| `K_REGULAR` | 20 | **40** | Modern NFL is more volatile (salary cap, free agency) |
| `K_PLAYOFFS` | 24 | **48** | Proportional to K_REGULAR |
| `REGRESSION_FACTOR` | 0.33 | **0.40** | Greater parity, teams change faster |
| `GLOBAL_MEAN` | 1505 | **1503.1** | Actual empirical mean of the system |

Brier Score improvement vs. FiveThirtyEight parameters: 0.0032 out-of-sample.

---

## Key design decisions

| Decision | Choice | Alternative discarded | Reason |
|---|---|---|---|
| ELO scope | REG + WC + DIV + CON + SB | REG only | Maximizes cross-season information |
| HOME_ADVANTAGE Super Bowl | 0 | 42.2 | Neutral field |
| Vegas in the model | Benchmark only | Model feature | Circular reasoning |
| `qb_name` | Not a feature | Include as dummy | ELO captures it indirectly |
| 2020 COVID | `sample_weight = 0.3` | Exclude | Keep N, reduce influence |
| Validation | Expanding walk-forward | Standard K-Fold | Avoids temporal leakage |
| `StandardScaler` | Fit on train only, per fold | Global | Avoids scale leakage |
| 2026 prediction engine | Pure ELO | Logistic regression | ELO beats LR and GB on Brier Score |
| `K_PLAYOFFS` | K × 1.2 | Independent calibration | Too few playoff games (~338) to estimate K separately |

### Predictive model — features

| Feature | Included | Reason |
|---|---|---|
| `elo_diff` | ✅ | Main feature · captures most of the predictive power |
| `home_rest` | ✅ | Home team days of rest |
| `away_rest` | ✅ | Away team days of rest |
| `div_game` | ✅ | Divisional rivalries have their own dynamics |
| `week_norm` | ✅ | `week / max_week` · normalizes the 16-to-17-game transition (2021) |
| `home_moneyline` | ❌ | External benchmark only · would be circular reasoning as a feature |
| `qb_name` | ❌ | ELO captures it indirectly through results |
| `temp` / `wind` | ❌ | High missing rate · weak signal vs. imputation cost |

---

## EDA findings

- **Dataset:** 7,276 games × 46 columns · seasons 1999–2025
- **Home win rate:** 56.3% overall · declining trend in the post-COVID era (2021–2025: ~53–54%)
- **2020 COVID:** structural outlier (home win rate ~49%) · `sample_weight = 0.3` in training
- **2021 transition:** 17-game season → `week_norm = week / max_week`
- **Missing Vegas data:** concentrated in 1999–2005 · benchmark applies to 2007–2025
- **Franchise unification:** OAK→LV, SD→LAC, STL→LA (normalized via `nflreadpy`)

---

## 2026 predictions — highlights

```
Top Super Bowl favorites:
  SEA: 12.8%  ← defending Super Bowl LX champion
  DEN:  7.8%
  NE:   7.4%
  HOU:  6.8%

Teams that fell the most vs. historical expectations:
  KC: 34.7% playoff odds · 1.6% Super Bowl  ← dynasty era in decline

Most competitive division:
  NFC West: SEA (80.9%) + LA (60.7%) + SF (56.0%) — three teams >56% playoff odds

Most likely Wild Card matchups:
  AFC: HOU (bye) · DEN vs LAC · NE vs JAX · BAL vs BUF
  NFC: SEA (bye) · PHI vs MIN · DET vs SF · ATL vs LA
```

---

## In-season weekly content (2026)

Unlike `simulation.py` (which simulates the full season assuming nothing has been played),
the in-season engine updates ELO **only with games already played** and computes each game's
win probability with the current rating. It reuses the calibrated functions in `src/elo.py`
(`expected_win_prob`, `update_elo`, `apply_season_regression`) and the same parameters.

Run every Tuesday (after Monday Night Football) with a single command:

1. **Data** — downloads the schedule with `nflreadpy`, recomputes ELO game by game and refreshes
   `data/processed/` (same files as Notebook 05)
2. **Metrics** — evaluates the model on played games (pick threshold configurable, default > 50%):
   record, Brier, log loss, calibration by confidence bucket, home/away picks, Vegas comparison
3. **Graphics** — the same images in two sizes on a graph-paper canvas:
   `size_1/` 1080×902 (feed post) and `size_2/` 1080×1920 (reel / story)

| # | Image | Content |
|---|---|---|
| 01 | `aciertos_por_semana` | Hits per week + cumulative |
| 02 | `elo_semana_XX` | ELO risers / fallers last week |
| 03 | `elo_temporada` | ELO risers / fallers since preseason |
| 04 | `top_elo` | Top 5 by ELO |
| XX | `picks_semana_XX` | Week picks in kickoff order, with logos |
| 06 | `el_mas_seguro` | Safest pick of the week |
| 07 | `la_moneda_al_aire` | Coin flip game |
| 08 | `la_sorpresa` | Upset candidate |
| 09 | `evolucion_elo` | Season ELO evolution (top 5 in color) |
| 10 | `evolucion_divisiones` | ELO evolution by division (all 32 teams) |
| — | `partidos/` | One story per game (record, logos, probability, comment) + `comentarios.md` |

Plus `resumen.md` (numbers for the script) and `evaluacion_partidos.csv`,
`metricas_semanales.csv`, `calibracion.csv` at the root of `reels/semana_XX/`.

### Season performance — weeks 1–7 (pick if prob. > 50%)

| Metric | Model (ELO) | Vegas favorite | Coin flip |
|---|---|---|---|
| Record | **60-46-1 (56.5%)** | 67-39-1 (63.1%) | — |
| Brier Score | 0.236 | 0.231 | 0.250 |
| Log loss | 0.668 | — | 0.693 |

The model agrees with the Vegas favorite in 82.2% of picks. Full weekly detail in
`reels/semana_XX/resumen.md`.

---

## Model 2 · ELO + absences (injuries)

A second model that runs **in parallel** with the calibrated ELO (Model 1). Same ELO
(HA = 42.2, K = 40/48, α = 0.40, μ = 1503.1) plus a pre-game adjustment for missing starters:

```
diff = R_home − R_away + HA − Σ_g δ_g · A_home,g + Σ_g δ_g · A_away,g
```

- `A_t,g` = expected missing starters of group g (QB, RB, WR/TE, OL, DEF) from nflreadpy:
  `load_injuries` (Out / Doubtful / Questionable), `load_rosters_weekly` (IR), `load_snap_counts`
  (who is a starter), `load_schedules` (starting QB). QB starter = week-1 starter.
- `δ_g` = ELO points per missing starter, fit by minimizing log loss on 2012–2021; groups kept
  only if they help in leave-one-season-out CV. Result: **QB ≈ 80, WR/TE ≈ 30**, RB/OL/DEF = 0.
- P(starter sits | status), 2012–2025: Out 99.97% · Doubtful 98.8% · Questionable 26.7%.

| Out-of-sample 2022–2025 (1,139 games) | Brier | Log loss | Accuracy |
|---|---|---|---|
| Model 1 · ELO | 0.2228 | 0.6376 | 64.2% |
| QB patch only (δ = 90) | 0.2197 | 0.6309 | 65.2% |
| **Model 2 · ELO + absences** (integrated, Friday report) | **0.2191** | **0.6300** | 64.8% |
| Model 2 (overlay) with Tuesday info (previous week's report) | 0.2195 | 0.6312 | 65.9% |
| Vegas | 0.2095 | 0.6075 | 67.6% |

Paired bootstrap: Model 2 beats Model 1 in 98% of resamples. Details and charts in
`Notebooks/07_elo_ausencias.ipynb`.

**Forecast ledger.** `weekly_content.py` freezes both models' forecasts before kickoff in
`data/processed/pronosticos/pronosticos_2026.csv` (append-only, timestamped). Evaluation uses
the frozen version, never a recomputation, so Model 2 is never graded with injury news it did
not have. Each `reels/semana_XX/` gets `comparacion_modelos.csv`, `ausencias_semana_XX.csv`
and a "Modelo 1 vs Modelo 2" section in `resumen.md`. Reel images still use Model 1.

```bash
python3 src/weekly_content.py                     # Tuesday: content + freeze both models
python3 src/weekly_content.py --solo-pronosticos  # Fri/Sat: re-freeze Model 2 with the final injury report
python3 src/weekly_content.py --sin-ausencias     # Model 1 only (previous behavior)
python3 src/compare_models.py --tabla             # cumulative comparison from the ledger
python3 src/compare_models.py --recalibrar        # re-estimate δ and absence probabilities
```

---

## Repository structure

```
nfl-probability-engine/
├── Notebooks/
│   ├── 01_exploratory_analysis.ipynb   ✅ EDA · parameter calibration
│   ├── 02_elo_system.ipynb             ✅ ELO system · validation
│   ├── 03_logistic_model.ipynb         ✅ Logistic regression vs. GB
│   ├── 04_monte_carlo.ipynb            ✅ 2026 season simulation
│   ├── 05_weekly_win_probabilities.ipynb ✅ In-season weekly ELO · per-game probabilities
│   └── 07_elo_ausencias.ipynb          ✅ Model 2 (ELO + absences) · audit and backtest
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       — load_nfl_data()
│   ├── elo.py               — full ELO system (computation, calibration, backtesting)
│   ├── model.py             — build_features, walk_forward_validation, train_final_model
│   ├── simulation.py        — run_monte_carlo, simulate_regular_season, playoff bracket
│   ├── visualization.py     — standings, bracket, per-team schedule, LinkedIn carousel
│   ├── weekly_probabilities.py — in-season ELO updates with real results only
│   ├── weekly_content.py    — weekly pipeline: data → metrics → reel images (CLI)
│   ├── availability.py      — missing starters per team-game (injuries, IR, snaps)
│   ├── elo_ausencias.py     — Model 2: ELO + absences (calibration, weekly snapshots)
│   ├── model_registry.py    — forecast ledger for Model 1 vs Model 2
│   └── compare_models.py    — CLI: --recalibrar, --backfill, --tabla
│
├── data/
│   └── processed/
│       ├── elo_history.csv            — game-by-game ELO 1999–2025
│       ├── elo_final_2025.csv         — end-of-2025 team ratings
│       ├── model_features.csv         — dataset with engineered features
│       ├── model_coefficients.csv     — final β coefficients
│       ├── predictions_2026.csv       — aggregated Monte Carlo results
│       ├── wins_distribution_2026.csv — win distribution per simulation
│       ├── game_results_2026.csv      — win probability per game
│       ├── weekly_predictions_2026.csv — in-season per-game ELO pre/post and probability
│       ├── elo_evolution_2026.csv     — week-by-week ELO per team
│       ├── weekly_predictions_ausencias_2026.csv — Model 2 per-game probabilities
│       ├── params_ausencias.json      — Model 2 δ, mode, absence probabilities
│       ├── backtest_ausencias.csv     — out-of-sample comparison
│       ├── ausencias_panel.csv        — missing starters per team-game 2012–2026
│       ├── pronosticos/               — frozen forecast ledger (both models)
│       └── weekly/                    — week_01_probabilities.csv … week_18_probabilities.csv
│   └── logos/                         — local cache of team logos (ARI.png … WAS.png)
│
├── results/
│   ├── regular_season_standings_2026.png
│   ├── wildcard_bracket_2026.png
│   └── carousel/                      — LinkedIn slides (slide_01.png … slide_10.png)
│
├── reels/
│   └── semana_XX/                     — one folder per week (semana_03 … semana_08)
│       ├── size_1/                    — 1080×902 images (+ partidos/)
│       ├── size_2/                    — 1080×1920 images (+ partidos/)
│       ├── resumen.md                 — weekly script with numbers and picks
│       └── *.csv                      — evaluation, weekly metrics, calibration
│
├── streamlit_app.py         ⏳ Pending (Phase 5)
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Project phases

| Phase | Notebook | Status | Key output |
|---|---|---|---|
| 1 · EDA | `01_exploratory_analysis.ipynb` | ✅ Complete | Calibrated parameters · Decision Log |
| 2 · ELO | `02_elo_system.ipynb` | ✅ Complete | `elo_history.csv` · `elo_final_2025.csv` · BS=0.2235 |
| 3 · Model | `03_logistic_model.ipynb` | ✅ Complete | β coefficients · BS=0.2248 · LR vs. GB comparison |
| 4 · Simulation | `04_monte_carlo.ipynb` | ✅ Complete | `predictions_2026.csv` · visualizations |
| In-season · Weekly | `05_weekly_win_probabilities.ipynb` + `src/weekly_content.py` | 🔄 Running weekly | `weekly_predictions_2026.csv` · `reels/semana_XX/` |

---

## Implemented functions

### `src/data_loader.py`
```python
load_nfl_data(seasons=range(1999, 2026))
# → clean DataFrame with normalized franchises and home_team_win
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
# → accepts any sklearn model; infers whether StandardScaler is needed
train_final_model(df, feature_cols)
```

### `src/simulation.py`
```python
# League constants
CONFERENCES, DIVISIONS, TEAM_CONF, TEAM_DIV

run_monte_carlo(schedule_df, final_elos_2025, home_advantage, k_regular,
                k_playoffs, regression_factor, global_mean,
                n_sims=10_000, seed=42)
# → results_df, wins_dist, game_results_df

simulate_regular_season(schedule_df, elos, home_advantage, k_regular, rng)
# → elos, records, game_results  (list of 0/1 per game)

determine_playoff_seeds(records, rng)
# → {'AFC': [s1..s7], 'NFC': [s1..s7]}

simulate_playoff_bracket(seeds, elos, home_advantage, k_playoffs, rng)
# → sb_winner, conf_champs, elos
```

### `src/visualization.py`
```python
# Internal helpers
_get_logo(team, size=(45,45))     # downloads, resizes and caches ESPN logos
_add_logo(ax, team, x, y, zoom)   # places a logo on a matplotlib axis

# Public functions
plot_regular_season_standings(results_df, bg_color='#1c1c1e', save_path=None)
plot_wildcard_bracket(results_df, bg_color='#1c1c1e', save_path=None)
plot_team_schedule(team, game_results_df, bg_color='#1c1c1e', save_path=None)
generate_linkedin_carousel(results_df, save_dir='../results/carousel', bg_color='#1c1c1e')
```

### `src/weekly_probabilities.py`
```python
build_weekly_snapshots(schedule_df, final_elos_prev_season, home_advantage,
                       k_regular, k_playoffs, regression_factor, global_mean)
# → games_df (ELO pre/post + probability per game), elo_evolution_df

current_week_probabilities(games_df)
# → games of the next unplayed week (or the last week if the season is over)

team_schedule_view(games_df, team)
# → one row per game played by the team
```

### `src/weekly_content.py`
```python
generar_contenido(season=2026, threshold=0.50, semana=None, actualizar=True,
                  out_dir=None, zona='America/Mexico_City', logos_dir=None,
                  top_n=5, historias=True, formatos=None)
# → dict with DataFrames (e.g. res["metricas"]) and paths of generated files

# Main steps
cargar_datos(season)                       # schedule with real scores + previous ELO
calcular_elo(schedule, final_elos)         # → build_weekly_snapshots
guardar_archivos(games_df, elo_evolution_df)
evaluar_partidos(games, threshold)         # per-game hits vs. model and Vegas
metricas_por_semana(ev) · calibracion(ev)
elegir_destacados(week_games)              # safest · coin flip · upset
graf_*(...)                                # one function per image
escribir_resumen(...) · escribir_comentarios(...)
```

---

## Visualization design system

```python
# Consistent palette across the project
BG_COLOR   = '#1c1c1e'    # Apple dark gray
CARD_BG    = '#2c2c2e'    # Apple secondary
AFC_COLOR  = '#013369'    # Official NFL AFC blue
NFC_COLOR  = '#D50A0A'    # Official NFL NFC red
GOLD_COLOR = '#f5c518'    # Gold accent (division leaders)

# Logos: downloaded from ESPN via nflreadpy.load_teams()
# Resized to 45×45px before OffsetImage with zoom=0.50
# ABBR_MAP = {'LA': 'LAR', 'JAX': 'JAC'}  ← abbreviation correction
```

### Weekly reels design system ("Black Sea")

```python
FONDO    = '#332E32'    # Black Sea background, always flat
PANEL    = '#3E383D'    # panels, bar tracks, highlighted rows
LINEA    = '#554E54'    # dividers and secondary bar segments
NEUTRO   = '#686167'    # neutral bar segment
TEXTO    = '#F6F2EE'    # primary text (warm white)
TEXTO2   = '#C2BBC0'    # secondary text
TENUE    = '#8E878C'    # footnotes, column headers
AZUL     = '#7778DE'    # Blueberry Pie: favorite / positive / what rises (text: #8F90EC)
ROJO     = '#F44631'    # Cherry Tomato: the lead / surprise / what falls (text: #F65A47)

# DejaVu Sans · flat design (no gradients, shadows or background grid)
# Story 1080×1920: content between y = 300 and y = 1780 px, 64 px side margins
# Footer: "— NFL · TOPIC · No es consejo de apuesta" + @samplesebs
```

---

## Evaluation metrics

```
Primary metric     : Brier Score  (probabilistic calibration)
Secondary metrics  : Log-Loss · Accuracy · Calibration Curve
Benchmark          : Vegas moneylines, vig-free (2010–2025)
Validation         : Expanding window (walk-forward)
    → Train 2010–2019 | Test 2020–2025 (ELO calibration)
```

---

## Tech stack

```
Python 3.11+
├── nflreadpy        — data source (replaces deprecated nfl_data_py)
├── pandas / numpy    — data manipulation
├── scikit-learn      — logistic regression · Gradient Boosting · metrics
├── scipy             — statistical testing
├── matplotlib        — visualizations
├── Pillow / requests — team logos (download, crop, cache)
├── tqdm              — Monte Carlo progress bar
└── streamlit         — interactive dashboard (Phase 5)
```

---

## Installation

```bash
git clone https://github.com/sebastianBP26/nfl-probability-engine.git
cd nfl-probability-engine
pip install -r requirements.txt
```

Run the notebooks in order:

```
01_exploratory_analysis.ipynb  →  02_elo_system.ipynb  →  03_logistic_model.ipynb  →  04_monte_carlo.ipynb
```

Weekly content (every Tuesday, from the repo root):

```bash
python src/weekly_content.py                    # automatic week
python src/weekly_content.py --formatos size_1  # only 1080x902 images
python src/weekly_content.py --threshold 0.6    # only count picks > 60%
python src/weekly_content.py --semana 5         # rebuild week 5 content
python src/weekly_content.py --sin-actualizar   # don't rewrite data/processed
```

IG channel stats (after `weekly_content.py`, so Model 2's forecasts are already frozen):

```bash
python src/canal_stats.py                       # automatic week
python src/canal_stats.py --semana 10           # rebuild week 10
python src/canal_stats.py --sin-partidos        # only the position slides
```

---

## IG channel · cumulative stats (2026)

Weekly slides for the Instagram channel with the season-to-date numbers (weeks 1 … N-1), in the same "Black Sea" design system as the reels. They are saved to `canal/semana_XX/`, in `size_2` (1080×1920) and `size_1` (1080×902).

| File | Content |
|---|---|
| `01_qb` … `06_k` | One slide per position: QB, RB, WR, TE, DEF and K. Each one has **cumulative leaders** (top 5), the **most consistent** players (top 5 by CV, among the better half of the position, with a p20–p80 range bar) and a **highlight pill**, e.g. "8/8 games with 7+ points". |
| `07_resumen` | A 2×3 dashboard with the leader and the most consistent player at each position. |
| `partidos/NN_AWAY_HOME` | One dashboard per game of the week, with five blocks: **Model 2 (Elo + absences)** win probability, Las Vegas spread and total, offense and defense side by side, key players (QB/RB1/WR1/TE1: mean ± SD and % of games with a TD) and the **key matchup**. The key matchup is the player whose position faces the most permissive opposing defense. |
| `tablas/*.csv` | The source of every number: `acumulado_{pos}.csv`, `equipos.csv` and `partidos.csv` (which includes Model 2's probability). |
| `resumen.md` | Supporting text for the channel posts. |

**Data sources:**
- `nflreadpy`: `load_player_stats`, `load_team_stats` and `load_schedules`.
- Forecasts come from the frozen forecast ledger (`model_registry.pronostico_vigente`, `modelo == "elo_ausencias"`). They are never recalculated, and Model 1 is never used as a fallback.

**Filters:**
- A game counts toward a player only with real volume: QB 15+ attempts, RB 6+ touches, WR 3+ targets, TE 2+ targets, K 1+ kick.
- To appear in a table, a player must have played at least 60% of their team's games.

The data and stats functions live in `src/player_stats.py`, which is shared with the Notebook 06 logic.

---

## Next steps

- [x] Explore in-season ELO updates as the 2026 season progresses
- [ ] Keep publishing weekly content through week 18 and playoffs

---

## References

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
  → [Internet Archive](https://archive.org/details/ratingofchesspla00unse)
- FiveThirtyEight NFL ELO Data & Code.
  → https://github.com/fivethirtyeight/data/tree/master/nfl-elo
