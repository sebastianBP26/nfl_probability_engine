# src/simulation.py

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Estructura de la liga 2026 ────────────────────────────────────────────────

CONFERENCES = {
    'AFC': ['BAL', 'CLE', 'PIT', 'CIN',
            'HOU', 'IND', 'JAX', 'TEN',
            'BUF', 'MIA', 'NE',  'NYJ',
            'KC',  'LV',  'LAC', 'DEN'],
    'NFC': ['CHI', 'DET', 'GB',  'MIN',
            'ATL', 'CAR', 'NO',  'TB',
            'DAL', 'NYG', 'PHI', 'WAS',
            'LA',  'SEA', 'SF',  'ARI'],
}

DIVISIONS = {
    'AFC_North': ['BAL', 'CLE', 'PIT', 'CIN'],
    'AFC_South': ['HOU', 'IND', 'JAX', 'TEN'],
    'AFC_East' : ['BUF', 'MIA', 'NE',  'NYJ'],
    'AFC_West' : ['KC',  'LV',  'LAC', 'DEN'],
    'NFC_North': ['CHI', 'DET', 'GB',  'MIN'],
    'NFC_South': ['ATL', 'CAR', 'NO',  'TB'],
    'NFC_East' : ['DAL', 'NYG', 'PHI', 'WAS'],
    'NFC_West' : ['LA',  'SEA', 'SF',  'ARI'],
}

TEAM_CONF = {t: c for c, teams in CONFERENCES.items() for t in teams}
TEAM_DIV  = {t: d for d, teams in DIVISIONS.items()  for t in teams}


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def _apply_regression(elos, regression_factor, global_mean):
    """Regresión a la media al inicio de la temporada."""
    return {
        t: (1 - regression_factor) * e + regression_factor * global_mean
        for t, e in elos.items()
    }


def _expected_prob(elo_home, elo_away, ha):
    """P(home wins) dada diferencia de ELO."""
    return 1 / (1 + 10 ** (-((elo_home - elo_away + ha) / 400)))


def _simulate_game(elo_home, elo_away, ha, k, rng):
    """
    Simula un partido y actualiza ELOs.

    Returns
    -------
    result       : int   — 1=gana local, 0=gana visitante
    elo_home_new : float
    elo_away_new : float
    """
    p      = _expected_prob(elo_home, elo_away, ha)
    result = int(rng.random() < p)
    delta  = k * (result - p)
    return result, elo_home + delta, elo_away - delta


# ── Función 1: temporada regular ─────────────────────────────────────────────

def simulate_regular_season(schedule_df, elos, home_advantage, k_regular, rng):
    """
    Simula una temporada regular completa partido a partido.

    Parameters
    ----------
    schedule_df    : pd.DataFrame — partidos REG 2026 ordenados por semana
    elos           : dict         — {team: elo_rating} al inicio de la temporada
    home_advantage : float
    k_regular      : float
    rng            : np.random.Generator

    Returns
    -------
    elos    : dict — ratings actualizados
    records : dict — {team: {wins, losses, div_wins, conf_wins}}
    """
    elos    = elos.copy()
    records = {
        t: {'wins': 0, 'losses': 0, 'div_wins': 0, 'conf_wins': 0}
        for t in elos
    }

    for g in schedule_df.itertuples(index=False):
        home = g.home_team
        away = g.away_team

        result, elos[home], elos[away] = _simulate_game(
            elos[home], elos[away], home_advantage, k_regular, rng
        )

        winner = home if result == 1 else away
        loser  = away if result == 1 else home

        records[winner]['wins']   += 1
        records[loser]['losses']  += 1

        # División y conferencia para tiebreakers
        if TEAM_DIV[home] == TEAM_DIV[away]:
            records[winner]['div_wins'] += 1

        if TEAM_CONF[home] == TEAM_CONF[away]:
            records[winner]['conf_wins'] += 1

    return elos, records


# ── Función 2: seeding de playoffs ───────────────────────────────────────────

def determine_playoff_seeds(records, rng):
    """
    Determina los 7 seeds de playoffs por conferencia.

    Reglas:
      - 4 campeones de división (seeds 1–4), ordenados por record
      - 3 wildcards (seeds 5–7), mejores records entre no-campeones
      - #1 seed tiene bye en Wild Card

    Tiebreaker: wins → div_wins → conf_wins → aleatorio

    Returns
    -------
    dict: {'AFC': [s1,...,s7], 'NFC': [s1,...,s7]}
    """
    def sort_key(t, use_div=False):
        return (
            records[t]['wins'],
            records[t]['div_wins'] if use_div else records[t]['conf_wins'],
            records[t]['conf_wins'],
            rng.random(),
        )

    seeds = {}

    for conf, all_teams in CONFERENCES.items():

        div_winners = []

        for div_name, div_teams in DIVISIONS.items():
            if not div_name.startswith(conf):
                continue
            winner = max(div_teams, key=lambda t: sort_key(t, use_div=True))
            div_winners.append(winner)

        # Seeds 1–4: campeones ordenados por record
        div_winners_sorted = sorted(
            div_winners,
            key=lambda t: sort_key(t),
            reverse=True
        )

        # Seeds 5–7: wildcards
        non_winners = [t for t in all_teams if t not in div_winners]
        wildcards   = sorted(
            non_winners,
            key=lambda t: sort_key(t),
            reverse=True
        )[:3]

        seeds[conf] = div_winners_sorted + wildcards

    return seeds


# ── Función 3: bracket de playoffs ───────────────────────────────────────────

def simulate_playoff_bracket(seeds, elos, home_advantage, k_playoffs, rng):
    """
    Simula el bracket completo de playoffs.

    Wild Card:   #2 vs #7, #3 vs #6, #4 vs #5  (#1 tiene bye)
    Divisional:  #1 vs lowest remaining, #2/#3 vs highest remaining
    Conference:  mejor seed es local
    Super Bowl:  cancha neutral (HA = 0)

    Returns
    -------
    sb_winner      : str  — campeón del Super Bowl
    conf_champs    : dict — {'AFC': team, 'NFC': team}
    elos           : dict — ratings finales
    """
    elos = elos.copy()
    conf_champs = {}

    for conf in ['AFC', 'NFC']:

        s = seeds[conf]   # [s1, s2, s3, s4, s5, s6, s7]

        # ── Wild Card ─────────────────────────────────────────────────────────
        # s1 tiene bye · s2 vs s7 · s3 vs s6 · s4 vs s5
        wc_survivors = {s[0]: 1}   # {team: seed_original}

        for seed_hi, seed_lo in [(1, 6), (2, 5), (3, 4)]:
            home, away = s[seed_hi], s[seed_lo]
            result, elos[home], elos[away] = _simulate_game(
                elos[home], elos[away], home_advantage, k_playoffs, rng
            )
            winner = home if result == 1 else away
            wc_survivors[winner] = seed_hi + 1   # seed 1-based

        # ── Divisional ────────────────────────────────────────────────────────
        # Re-seedear supervivientes por seed original
        sorted_surv = sorted(wc_survivors.keys(),
                             key=lambda t: wc_survivors[t])
        # sorted_surv = [mejor_seed, ..., peor_seed] (4 equipos)

        # #1 vs #4_seed_survivor, #2_seed vs #3_seed_survivor
        div_matchups = [
            (sorted_surv[0], sorted_surv[3]),
            (sorted_surv[1], sorted_surv[2]),
        ]

        div_survivors = {}
        for home, away in div_matchups:
            result, elos[home], elos[away] = _simulate_game(
                elos[home], elos[away], home_advantage, k_playoffs, rng
            )
            winner = home if result == 1 else away
            div_survivors[winner] = wc_survivors[winner]

        # ── Conference Championship ────────────────────────────────────────────
        finalists  = list(div_survivors.keys())
        home_conf  = min(finalists, key=lambda t: div_survivors[t])
        away_conf  = max(finalists, key=lambda t: div_survivors[t])

        result, elos[home_conf], elos[away_conf] = _simulate_game(
            elos[home_conf], elos[away_conf], home_advantage, k_playoffs, rng
        )
        conf_champs[conf] = home_conf if result == 1 else away_conf

    # ── Super Bowl ─────────────────────────────────────────────────────────────
    # Cancha neutral → HA = 0
    # AFC es "local" por convención del dataset
    afc = conf_champs['AFC']
    nfc = conf_champs['NFC']

    result, elos[afc], elos[nfc] = _simulate_game(
        elos[afc], elos[nfc], ha=0, k=k_playoffs, rng=rng
    )
    sb_winner = afc if result == 1 else nfc

    return sb_winner, conf_champs, elos


# ── Función 4: Monte Carlo ─────────────────────────────────────────────────────

def run_monte_carlo(schedule_df, final_elos_2025,
                    home_advantage, k_regular, k_playoffs,
                    regression_factor, global_mean,
                    n_sims=10_000, seed=42):
    """
    Corre N simulaciones de la temporada 2026 completa.

    Parameters
    ----------
    schedule_df      : pd.DataFrame — partidos REG 2026
    final_elos_2025  : dict         — {team: elo} al cierre de 2025
    home_advantage   : float        — HOME_ADVANTAGE calibrado
    k_regular        : float        — K factor temporada regular
    k_playoffs       : float        — K factor postemporada
    regression_factor: float        — alpha de regresión entre temporadas
    global_mean      : float        — mu global
    n_sims           : int          — número de simulaciones
    seed             : int          — semilla para reproducibilidad

    Returns
    -------
    results_df : pd.DataFrame — métricas agregadas por equipo
    wins_dist  : pd.DataFrame — distribución de wins por simulación
    """
    rng = np.random.default_rng(seed)

    # ELOs 2026 tras regresión de offseason
    elos_2026 = _apply_regression(final_elos_2025, regression_factor, global_mean)

    # Preparar schedule
    sched = (
        schedule_df[schedule_df['game_type'] == 'REG']
        .sort_values('week')
        .reset_index(drop=True)
    )

    teams = sorted(elos_2026.keys())

    # Acumuladores
    wins_acc       = {t: np.zeros(n_sims, dtype=np.int8) for t in teams}
    playoffs_acc   = {t: 0 for t in teams}
    div_win_acc    = {t: 0 for t in teams}
    conf_champ_acc = {t: 0 for t in teams}
    sb_win_acc     = {t: 0 for t in teams}

    for sim in tqdm(range(n_sims), desc='Monte Carlo 2026'):

        # Temporada regular
        elos_sim, records = simulate_regular_season(
            sched, elos_2026, home_advantage, k_regular, rng
        )

        for t in teams:
            wins_acc[t][sim] = records[t]['wins']

        # Seeding
        seeds = determine_playoff_seeds(records, rng)

        for conf in ['AFC', 'NFC']:
            for i, t in enumerate(seeds[conf]):
                playoffs_acc[t] += 1
                if i < 4:
                    div_win_acc[t] += 1

        # Bracket de playoffs
        sb_winner, conf_champs, _ = simulate_playoff_bracket(
            seeds, elos_sim, home_advantage, k_playoffs, rng
        )

        conf_champ_acc[conf_champs['AFC']] += 1
        conf_champ_acc[conf_champs['NFC']] += 1
        sb_win_acc[sb_winner]              += 1

    # ── Resultados agregados ──────────────────────────────────────────────────
    results_df = pd.DataFrame({
        'team'          : teams,
        'conf'          : [TEAM_CONF[t] for t in teams],
        'division'      : [TEAM_DIV[t]  for t in teams],
        'avg_wins'      : [round(wins_acc[t].mean(), 2) for t in teams],
        'std_wins'      : [round(wins_acc[t].std(),  2) for t in teams],
        'pct_playoffs'  : [round(playoffs_acc[t]   / n_sims * 100, 1) for t in teams],
        'pct_div_winner': [round(div_win_acc[t]    / n_sims * 100, 1) for t in teams],
        'pct_conf_champ': [round(conf_champ_acc[t] / n_sims * 100, 1) for t in teams],
        'pct_sb_winner' : [round(sb_win_acc[t]     / n_sims * 100, 1) for t in teams],
    }).sort_values('pct_playoffs', ascending=False).reset_index(drop=True)

    wins_dist = pd.DataFrame({t: wins_acc[t] for t in teams})

    return results_df, wins_dist