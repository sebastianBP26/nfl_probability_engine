# src/weekly_probabilities.py
"""
Motor de probabilidades semanales in-season.

A diferencia de simulation.py (que corre Monte Carlo sobre la temporada
completa asumiendo que nada se ha jugado), este módulo actualiza los ratings
ELO partido a partido usando SOLO resultados reales ya ocurridos, y calcula
la probabilidad de victoria de cada juego con el ELO vigente a esa fecha.

Lógica:
    Semana 1 -> ELOs de cierre de temporada anterior + regresión de temporada
    Semana N -> ELOs actualizados con los resultados reales de las semanas 1..N-1

Se apoya en las funciones ya calibradas de src/elo.py (expected_win_prob,
update_elo, apply_season_regression) para no duplicar lógica.
"""

import pandas as pd

from src.elo import expected_win_prob, update_elo, apply_season_regression


def build_weekly_snapshots(
    schedule_df,
    final_elos_prev_season,
    home_advantage,
    k_regular,
    k_playoffs,
    regression_factor,
    global_mean,
):
    """
    Recorre el calendario de la temporada semana a semana, actualizando los
    ratings ELO únicamente con los partidos que ya tienen marcador
    (home_score / away_score no nulos). Los partidos futuros se dejan con el
    ELO más reciente disponible (el de la última semana jugada).

    Parameters
    ----------
    schedule_df             : pd.DataFrame — calendario de la temporada
                               (output de nfl.load_schedules(seasons=[YYYY]))
    final_elos_prev_season   : dict — {team: elo} al cierre de la temporada anterior
    home_advantage           : float
    k_regular                : float
    k_playoffs               : float
    regression_factor        : float
    global_mean               : float

    Returns
    -------
    games_df         : pd.DataFrame — un renglón por partido con ELOs pre/post,
                        probabilidad de victoria del local y resultado real
                        (si ya se jugó).
    elo_evolution_df  : pd.DataFrame — un renglón por (team, week) con el ELO
                        del equipo AL INICIO de esa semana. Útil para graficar
                        trayectorias de ELO por temporada.
    """
    game_type_k = {
        "REG": k_regular,
        "WC": k_playoffs,
        "DIV": k_playoffs,
        "CON": k_playoffs,
        "SB": k_playoffs,
    }

    sort_cols = ["week"]
    if "gameday" in schedule_df.columns:
        sort_cols = ["week", "gameday"]

    sched = (
        schedule_df.dropna(subset=["week"])
        .sort_values(sort_cols)
        .reset_index(drop=True)
        .copy()
    )

    elos = apply_season_regression(final_elos_prev_season, regression_factor, global_mean)

    game_records = []
    elo_records = []

    for week in sorted(sched["week"].unique()):
        week_games = sched[sched["week"] == week]

        # Snapshot del ELO de cada equipo ANTES de que se juegue esta semana
        for team, elo in elos.items():
            elo_records.append({"week": int(week), "team": team, "elo_pre_week": round(elo, 2)})

        for g in week_games.itertuples(index=False):
            home, away = g.home_team, g.away_team
            game_type = g.game_type
            ha = 0.0 if game_type == "SB" else home_advantage
            k = game_type_k.get(game_type, k_regular)

            elo_home_pre = elos[home]
            elo_away_pre = elos[away]
            p_home = expected_win_prob(elo_home_pre, elo_away_pre, ha)

            played = pd.notna(g.home_score) and pd.notna(g.away_score)
            result = None
            elo_home_post, elo_away_post = elo_home_pre, elo_away_pre

            if played:
                result = int(g.home_score > g.away_score)
                elo_home_post, elo_away_post = update_elo(
                    elo_home_pre, elo_away_pre, result, k, ha
                )
                elos[home] = elo_home_post
                elos[away] = elo_away_post

            game_records.append(
                {
                    "season": g.season,
                    "week": int(week),
                    "game_type": game_type,
                    "home_team": home,
                    "away_team": away,
                    "elo_home_pre": round(elo_home_pre, 2),
                    "elo_away_pre": round(elo_away_pre, 2),
                    "p_home_pre": round(p_home, 4),
                    "p_away_pre": round(1 - p_home, 4),
                    "played": played,
                    "home_score": g.home_score,
                    "away_score": g.away_score,
                    "result": result,
                    "elo_home_post": round(elo_home_post, 2),
                    "elo_away_post": round(elo_away_post, 2),
                }
            )

    games_df = pd.DataFrame(game_records)
    elo_evolution_df = pd.DataFrame(elo_records)
    return games_df, elo_evolution_df


def current_week_probabilities(games_df):
    """
    Devuelve los partidos de la próxima semana aún no jugada. Si toda la
    temporada ya se jugó, regresa la última semana disponible.
    """
    pending = games_df[~games_df["played"]]
    if pending.empty:
        return games_df[games_df["week"] == games_df["week"].max()]
    next_week = pending["week"].min()
    return games_df[games_df["week"] == next_week]


def team_schedule_view(games_df, team):
    """
    Reordena games_df a nivel equipo: un renglón por partido jugado por
    `team`, indicando si jugó de local/visitante, su probabilidad de
    victoria pre-partido y el resultado real si ya se jugó.
    """
    home_rows = games_df[games_df["home_team"] == team].copy()
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["is_home"] = True
    home_rows["team_win_prob"] = home_rows["p_home_pre"]
    home_rows["team_won"] = home_rows["result"]

    away_rows = games_df[games_df["away_team"] == team].copy()
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["is_home"] = False
    away_rows["team_win_prob"] = away_rows["p_away_pre"]
    away_rows["team_won"] = away_rows["result"].apply(
        lambda r: None if r is None else 1 - r
    )

    cols = [
        "season", "week", "game_type", "opponent", "is_home",
        "team_win_prob", "played", "team_won",
    ]
    out = pd.concat([home_rows[cols], away_rows[cols]], ignore_index=True)
    return out.sort_values("week").reset_index(drop=True)
