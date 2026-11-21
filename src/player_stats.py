# src/player_stats.py
"""
Estadísticas por jugador y por equipo (nflreadpy) · acumulado de la temporada
============================================================================

Lo que usa el Notebook 06 (EDA para jugadores constantes) y src/canal_stats.py
(láminas del canal de IG). Fuentes:

    nfl.load_player_stats(summary_level="week")  una fila por jugador x partido
        https://nflreadr.nflverse.com/articles/dictionary_player_stats.html
    nfl.load_team_stats(summary_level="week")    una fila por equipo x partido
    nfl.load_schedules()                         marcadores, spread y total de Las Vegas

Convenciones:
    - Solo temporada regular (season_type == "REG").
    - Un partido cuenta para un jugador solo si tuvo volumen real (VOLUMEN_MINIMO).
    - std con ddof=1 · cv = std / media · piso = media - std.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nflreadpy as nfl

# ── Métricas ──────────────────────────────────────────────────────────────────
METRICAS_POSICION = {
    "QB": ["passing_yards", "passing_tds", "completions", "attempts",
           "passing_interceptions", "rushing_yards", "pass_rush_yards", "fantasy_points"],
    "RB": ["rushing_yards", "carries", "rushing_tds", "receptions",
           "receiving_yards", "scrimmage_yards", "total_tds", "fantasy_points_ppr"],
    "WR": ["receiving_yards", "receptions", "targets", "receiving_tds",
           "receiving_air_yards", "total_tds", "fantasy_points_ppr"],
    "TE": ["receiving_yards", "receptions", "targets", "receiving_tds",
           "total_tds", "fantasy_points_ppr"],
    "K": ["fg_made", "fg_att", "fg_long", "pat_made", "pat_att", "puntos_k"],
}

# Volumen mínimo POR PARTIDO para que ese partido cuente (filtra apariciones marginales)
VOLUMEN_MINIMO = {
    "QB": ("attempts", 15),
    "RB": ("touches", 6),
    "WR": ("targets", 3),
    "TE": ("targets", 2),
    "K": ("kick_att", 1),
}

NOMBRES = {
    "passing_yards": "Yardas por pase", "passing_tds": "Pases de TD",
    "completions": "Pases completos", "attempts": "Pases intentados",
    "passing_interceptions": "Intercepciones", "rushing_yards": "Yardas por tierra",
    "pass_rush_yards": "Yardas pase + tierra", "fantasy_points": "Fantasy points",
    "carries": "Acarreos", "rushing_tds": "TD por tierra", "receptions": "Recepciones",
    "receiving_yards": "Yardas recibidas", "scrimmage_yards": "Yardas totales (tierra + aire)",
    "total_tds": "TD totales", "fantasy_points_ppr": "Fantasy points PPR",
    "targets": "Targets", "receiving_tds": "TD por recepción",
    "receiving_air_yards": "Air yards", "touches": "Toques",
    "fg_made": "Goles de campo", "puntos_k": "Puntos del pateador",
    "pts_permitidos": "Puntos permitidos", "yds_permitidas": "Yardas permitidas",
    "capturas": "Capturas", "entregas": "Entregas forzadas",
}


def nombre(metrica):
    return NOMBRES.get(metrica, metrica)


# ── Jugadores ─────────────────────────────────────────────────────────────────
def agregar_metricas_derivadas(df):
    """Columnas combinadas que suelen usarse como prop."""
    df = df.copy()
    num = lambda c: df[c].fillna(0) if c in df else 0
    df["scrimmage_yards"] = num("rushing_yards") + num("receiving_yards")
    df["total_tds"] = num("rushing_tds") + num("receiving_tds")
    df["touches"] = num("carries") + num("receptions")
    df["pass_rush_yards"] = num("passing_yards") + num("rushing_yards")
    df["kick_att"] = num("fg_att") + num("pat_att")
    df["puntos_k"] = 3 * num("fg_made") + num("pat_made")
    return df


def cargar_player_stats(temporadas=None, semanas=None, season_type="REG"):
    """
    Estadísticas por partido con nflreadpy, filtradas por fechas.

    temporadas  : None (actual) | int | list[int]
    semanas     : None (todas) | (inicio, fin) inclusivo | list[int]
    season_type : 'REG' | 'POST' | None (ambas)
    """
    df = nfl.load_player_stats(seasons=temporadas, summary_level="week").to_pandas()
    if season_type is not None:
        df = df[df["season_type"] == season_type]
    if semanas is not None:
        df = df[df["week"].between(*semanas)] if isinstance(semanas, tuple) \
            else df[df["week"].isin(semanas)]
    return agregar_metricas_derivadas(df).reset_index(drop=True)


def datos_posicion(df, posicion, volumen_minimo=VOLUMEN_MINIMO, ultimas_n=None):
    """Renglones (jugador x partido) de una posición, solo partidos con volumen real."""
    d = df[df["position"] == posicion].copy()
    if posicion in volumen_minimo:
        col, minimo = volumen_minimo[posicion]
        d = d[d[col].fillna(0) >= minimo]
    d = d.sort_values(["player_id", "season", "week"])
    if ultimas_n is not None:
        d = d.groupby("player_id", group_keys=False).tail(ultimas_n)
    metricas = [m for m in METRICAS_POSICION.get(posicion, []) if m in d]
    d[metricas] = d[metricas].fillna(0)
    return d.reset_index(drop=True)


def resumen_jugadores(df_pos, metrica, min_partidos=3, id_col="player_id",
                      nombre_col="player_display_name"):
    """
    Una fila por jugador con la distribución de `metrica` por partido:
    partidos, total, media, std, cv, piso, mediana, p20, p80, mínimo y máximo.
    """
    g = df_pos.groupby([id_col, nombre_col])
    res = g[metrica].agg(partidos="count", total="sum", media="mean", std="std",
                         mediana="median", minimo="min", maximo="max",
                         p20=lambda s: s.quantile(0.2),
                         p80=lambda s: s.quantile(0.8)).reset_index()
    res["equipo"] = g["team"].last().values  # equipo más reciente
    res = res[res["partidos"] >= min_partidos].copy()
    res["std"] = res["std"].fillna(0)
    res["cv"] = np.where(res["media"] > 0, res["std"] / res["media"], np.nan)
    res["piso"] = res["media"] - res["std"]
    return (res.rename(columns={nombre_col: "jugador"})
            .sort_values("media", ascending=False).reset_index(drop=True))


# ── Equipos ───────────────────────────────────────────────────────────────────
def cargar_calendario(temporada):
    return nfl.load_schedules(seasons=[temporada]).to_pandas()


def partidos_equipo(temporada, semana_max, calendario=None):
    """
    Una fila por equipo x partido (REG, week <= semana_max) con su ataque y su
    defensa. La defensa se mide con la fila del rival en load_team_stats:

        pts, yds                 puntos y yardas (pase + tierra) que anotó / ganó
        pts_permitidos           puntos del rival (load_schedules)
        yds_permitidas           yardas del rival
        capturas                 def_sacks
        entregas                 def_interceptions + balones sueltos que perdió el rival
    """
    t = nfl.load_team_stats(seasons=[temporada], summary_level="week").to_pandas()
    t = t[(t["season_type"] == "REG") & (t["week"] <= semana_max)].copy()
    t["yds"] = t["passing_yards"].fillna(0) + t["rushing_yards"].fillna(0)
    t["perdidas_balon"] = t["fumbles_lost_total"].fillna(0)

    cal = calendario if calendario is not None else cargar_calendario(temporada)
    cal = cal[cal["home_score"].notna()]
    marcador = pd.concat([
        cal[["game_id", "home_team", "home_score", "away_score"]]
        .rename(columns={"home_team": "team", "home_score": "pts", "away_score": "pts_permitidos"}),
        cal[["game_id", "away_team", "away_score", "home_score"]]
        .rename(columns={"away_team": "team", "away_score": "pts", "home_score": "pts_permitidos"}),
    ])
    rival = t[["game_id", "team", "yds", "perdidas_balon"]].rename(
        columns={"team": "opponent_team", "yds": "yds_permitidas",
                 "perdidas_balon": "balones_rival"})
    d = (t.merge(rival, on=["game_id", "opponent_team"], how="left")
          .merge(marcador, on=["game_id", "team"], how="inner"))
    d["capturas"] = d["def_sacks"].fillna(0)
    d["entregas"] = d["def_interceptions"].fillna(0) + d["balones_rival"].fillna(0)
    cols = ["season", "week", "game_id", "team", "opponent_team", "pts", "yds",
            "pts_permitidos", "yds_permitidas", "capturas", "entregas"]
    return d[cols].sort_values(["team", "week"]).reset_index(drop=True)


def resumen_equipos(pe):
    """Promedios por partido de cada equipo (para el lado a lado de cada duelo)."""
    g = pe.groupby("team")
    res = g[["pts", "yds", "pts_permitidos", "yds_permitidas", "capturas", "entregas"]].mean()
    res["partidos"] = g.size()
    res["capturas_total"] = g["capturas"].sum()
    res["entregas_total"] = g["entregas"].sum()
    return res


def factor_defensa(df, posicion, metrica):
    """
    Lo que permite cada defensa por partido en `metrica` a una posición, entre
    el promedio de la liga (1.15 = permite 15% más que el promedio).
    """
    d = df[df["position"] == posicion]
    por_partido = d.groupby(["opponent_team", "game_id"])[metrica].sum()
    permitido = por_partido.groupby(level="opponent_team").mean()
    return permitido / permitido.mean()
