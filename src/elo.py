import itertools
import pandas as pd
import numpy as np
import nflreadpy as nfl



# Function to load and preprocess NFL data for given seasons
def load_nfl_data(seasons, cols_to_drop = None, franchise_map = None):
    
    if cols_to_drop is None:
        cols_to_drop = ['nfl_detail_id', 'pff', 'ftn', 'away_spread_odds', 'home_spread_odds','over_odds', 'under_odds']

    if franchise_map is None:
        franchise_map = {
                # Rams
                "STL": "LA",
                "LA": "LA",
                "LAR": "LA",

                # Raiders
                "OAK": "LV",
                "LV": "LV",
                "LVR": "LV",

                # Chargers
                "SD": "LAC",
                "LAC": "LAC"
            }

    df = nfl.load_schedules(seasons=list(range(1999, 2026))).to_pandas()
    df = df.drop(columns=cols_to_drop)
    df["home_team"] = df["home_team"].replace(franchise_map)
    df["away_team"] = df["away_team"].replace(franchise_map)
    df['home_team_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    return df

ELO_INICIAL    = 1500

# ── Función 1: HOME_ADVANTAGE ─────────────────────────────────────────────────
def estimate_home_advantage(df, last_n_seasons=None, game_types=['REG']):
    """
        Objetivo: Calibrar la ventaja de local (HOME_ADVANTAGE) a partir del histórico de partidos.
        La ventaja de local se define como la cantidad de puntos ELO que se le suman al rating del equipo local para calcular la probabilidad de victoria.
        
        Si Phome = Paway = 0.5, entonces HA = 0 (no hay ventaja de local). Despejando de la fórmula original HA tenemos:
        HA = -400 * log10((1 / P_home) - 1)

        Input:
            df             : pd.DataFrame — de resultados
            last_n_seasons : int         — si se quiere usar solo un número limitado de temporadas
            game_types     : list[str]   — tipos de partidos a considerar (REG, WC, SB, etc.)

        Output:
            dict — {
                'home_advantage': float — puntos ELO de ventaja local
                'home_win_rate' : float — tasa de victorias del local
                'n_games'       : int   — número de partidos usados para el cálculo
                'seasons'       : tuple — (temporada mínima, temporada máxima) de los partidos usados
            }
    """
    
    mask = df['game_type'].isin(game_types)

    if last_n_seasons is not None:
        max_season = df['season'].max()
        min_season = max_season - last_n_seasons + 1
        mask = mask & (df['season'] >= min_season)

    subset = df[mask]
    observed_rate = subset['home_team_win'].mean()
    ha = -400 * np.log10((1 / observed_rate) - 1)

    return {
        'home_advantage': round(ha, 1),
        'home_win_rate' : round(observed_rate, 4),
        'n_games'       : len(subset),
        'seasons'       : (subset['season'].min(), subset['season'].max())
    }


# ── Función 2: probabilidad esperada ──────────────────────────────────────────

def expected_win_prob(elo_home, elo_away, home_advantage):
    """
    Calcula la probabilidad de victoria del equipo local. Con base en la fórmula de ELO:

       * P_home = 1 / (1 + 10^(-((R_home - R_away + HA) / 400)))

    Parameters
    ----------
    elo_home       : float — rating ELO del equipo local
    elo_away       : float — rating ELO del equipo visitante
    home_advantage : float — puntos ELO de ventaja local (0 para SB)

    Returns
    -------
    float — probabilidad de victoria del local [0, 1]
    """
    diff = (elo_home - elo_away + home_advantage) / 400
    return 1 / (1 + 10 ** (-diff))


# ── Función 3: actualización de ratings ───────────────────────────────────────

def update_elo(elo_home, elo_away, result, k, home_advantage):
    """
    Actualiza los ratings ELO después de un partido.

    New_ELO_home = ELO_home + K * (S - P_home)
    New_ELO_away = ELO_away + K * ((1-S) - (1-P_home))

    Parameters
    ----------
    elo_home       : float — rating ELO del equipo local pre-partido
    elo_away       : float — rating ELO del equipo visitante pre-partido
    result         : int   — 1 si ganó local, 0 si ganó visitante
    k              : float — K-factor del partido | Su valor será estimado con los datos históricos.
    home_advantage : float — puntos ELO de ventaja local (0 para SB)

    Returns
    -------
    tuple (new_elo_home, new_elo_away)
    """
    p_home = expected_win_prob(elo_home, elo_away, home_advantage)

    delta = k * (result - p_home)

    return (
        round(elo_home + delta, 4),
        round(elo_away - delta, 4),
    )


# ── Función 4: regresión entre temporadas ─────────────────────────────────────

def apply_season_regression(elo_ratings, regression_factor, global_mean):
    """
    Jala todos los ratings hacia la media global al inicio de cada temporada.
    ELO_inicio = (1 - alpha) × ELO_final + alpha × mu
    """
    return {
        team: round((1 - regression_factor) * elo + regression_factor * global_mean, 4)
        for team, elo in elo_ratings.items()
    }


# ── Función 5: sistema completo ───────────────────────────────────────────────

def run_elo_system(df, home_advantage, regression_factor = 0.33, global_mean = 1505, k_regular = 20, k_playoffs = 24):
    """
    Corre el sistema ELO sobre todo el histórico partido a partido.

    Parameters
    ----------
    df                : pd.DataFrame — schedules limpio
    home_advantage    : float        — puntos ELO de ventaja local
    regression_factor : float        — alpha
    global_mean       : float        — mu
    k_regular         : int          — K factor temporada regular
    k_playoffs        : int          — K factor postemporada
    """
    game_type_k = {
        'REG': k_regular,
        'WC' : k_playoffs,
        'DIV': k_playoffs,
        'CON': k_playoffs,
        'SB' : k_playoffs,
    }

    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    teams       = set(df['home_team']).union(set(df['away_team']))
    elo_ratings = {team: ELO_INICIAL for team in teams}

    records        = []
    current_season = None

    for _, row in df.iterrows():

        season    = row['season']
        game_type = row['game_type']
        home_team = row['home_team']
        away_team = row['away_team']
        result    = row['home_team_win']

        if season != current_season:
            if current_season is not None:
                elo_ratings = apply_season_regression(
                    elo_ratings, regression_factor, global_mean
                )
            current_season = season

        k  = game_type_k[game_type]
        ha = 0.0 if game_type == 'SB' else home_advantage

        elo_home_pre = elo_ratings[home_team]
        elo_away_pre = elo_ratings[away_team]
        p_home       = expected_win_prob(elo_home_pre, elo_away_pre, ha)

        elo_home_post, elo_away_post = update_elo(
            elo_home_pre, elo_away_pre, result, k, ha
        )

        elo_ratings[home_team] = elo_home_post
        elo_ratings[away_team] = elo_away_post

        records.append({
            'season'        : season,
            'week'          : row['week'],
            'game_type'     : game_type,
            'home_team'     : home_team,
            'away_team'     : away_team,
            'elo_home_pre'  : elo_home_pre,
            'elo_away_pre'  : elo_away_pre,
            'home_advantage': ha,
            'p_home'        : round(p_home, 4),
            'result'        : result,
            'elo_home_post' : elo_home_post,
            'elo_away_post' : elo_away_post,
        })

    history_df = pd.DataFrame(records)
    final_elos = elo_ratings

    return history_df, final_elos

def backtest_home_advantage(df, ha_candidates, test_seasons, game_types=['REG']):
    """
    Compara distintos valores de HOME_ADVANTAGE usando Brier Score.

    Parameters
    ----------
    df            : pd.DataFrame — schedules limpio
    ha_candidates : dict         — {'nombre': valor_HA}
    test_seasons  : list         — temporadas de evaluación
    game_types    : list         — tipos de partido a evaluar

    Returns
    -------
    pd.DataFrame con Brier Score por configuración, ordenado de mejor a peor
    """
    results = []

    for name, ha in ha_candidates.items():

        # Correr ELO completo con este HA
        history_df, _ = run_elo_system(df, home_advantage=ha)

        # Filtrar período de test
        test_mask = (
            history_df['season'].isin(test_seasons) &
            history_df['game_type'].isin(game_types)
        )
        test_df = history_df[test_mask]

        # Brier Score
        bs = np.mean((test_df['p_home'] - test_df['result']) ** 2)

        results.append({
            'config'        : name,
            'home_advantage': ha,
            'brier_score'   : round(bs, 4),
            'n_games'       : len(test_df),
            'seasons'       : f"{min(test_seasons)}–{max(test_seasons)}"
        })

    return pd.DataFrame(results).sort_values('brier_score').reset_index(drop=True)

def calibrate_global_mean(df, home_advantage, regression_factor=0.33):
    """
    Calcula la media global empírica del sistema ELO.

    Aunque todos los equipos arrancan en 1500, la media puede derivar
    ligeramente por la regresión entre temporadas y los equipos de expansión.
    Esta función corre el sistema una vez y calcula la media real.

    Returns
    -------
    float — media empírica de todos los ratings ELO pre-partido
    """
    history_df, _ = run_elo_system(
        df,
        home_advantage=home_advantage,
        regression_factor=regression_factor,
        global_mean=1500    # valor neutro para el primer cálculo
    )

    all_elos = pd.concat([
        history_df['elo_home_pre'],
        history_df['elo_away_pre']
    ])

    mu = all_elos.mean()
    print(f"Media empírica calculada: {mu:.2f}  (referencia FiveThirtyEight: 1505)")
    return round(mu, 1)

def grid_search_elo(df, k_values, alphas, test_seasons, home_advantage, global_mean):
    """
    Busca la combinación óptima de K y alpha simultáneamente.
    """
    results = []

    combos = list(itertools.product(k_values, alphas))
    print(f"Evaluando {len(combos)} combinaciones...")

    for k, alpha in combos:
        history_df, _ = run_elo_system(
            df,
            home_advantage=home_advantage,
            regression_factor=alpha,
            global_mean=global_mean,
            k_regular=k,
            k_playoffs=round(k * 1.2)
        )

        test = history_df[
            history_df['season'].isin(test_seasons) &
            history_df['game_type'].isin(['REG'])
        ]

        bs = np.mean((test['p_home'] - test['result']) ** 2)
        results.append({
            'k_regular'  : k,
            'k_playoffs' : round(k * 1.2),
            'alpha'      : alpha,
            'brier_score': round(bs, 4),
        })

    return (
        pd.DataFrame(results)
          .sort_values('brier_score')
          .reset_index(drop=True)
    )