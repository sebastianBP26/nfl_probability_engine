# src/model_registry.py
"""
Libro de pronósticos · Modelo 1 (Elo base) vs Modelo 2 (Elo + ausencias)
=======================================================================

Cada vez que se corre el flujo semanal, los pronósticos de la semana se
CONGELAN aquí con fecha y hora. La evaluación usa lo congelado, no un
recálculo: el modelo 2 depende del reporte de lesiones del momento y
recalcularlo después usaría información que no existía al publicar.

Archivo: data/processed/pronosticos/pronosticos_<season>.csv (solo se agregan
renglones; nunca se borran ni se sobreescriben).

    modelo           elo_base | elo_ausencias
    origen           semanal (martes) | actualizacion (--solo-pronosticos)
                     | backfill (semanas ya jugadas, reconstruidas)
    generado_en      fecha-hora UTC en que se generó
    p_home, favorito, p_fav, ajuste_elo_*, ausentes_*, fuente_reporte ...

pronostico_vigente() toma, para cada partido y modelo, la última versión
generada ANTES del kickoff: esa es la que cuenta en la comparación.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIBRO_DIR = ROOT / "data" / "processed" / "pronosticos"

MODELOS = {
    "elo_base": "Modelo 1 · Elo",
    "elo_ausencias": "Modelo 2 · Elo + ausencias",
}

COLUMNAS = [
    "modelo", "origen", "generado_en", "season", "week", "game_id", "kickoff",
    "home_team", "away_team", "p_home", "p_away", "favorito", "p_fav",
    "elo_home_pre", "elo_away_pre", "ajuste_elo_home", "ajuste_elo_away",
    "ausentes_home", "ausentes_away", "fuente_reporte", "params",
]


def ruta_libro(season):
    return LIBRO_DIR / f"pronosticos_{season}.csv"


def cargar_libro(season):
    ruta = ruta_libro(season)
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS)
    return pd.read_csv(ruta, dtype={"ausentes_home": str, "ausentes_away": str})


def _filas(games, modelo, origen, generado_en, params):
    """games: salida de enriquecer() (base) o del modelo 2 enriquecida."""
    g = games.copy()
    for c in ("ajuste_elo_home", "ajuste_elo_away"):
        if c not in g:
            g[c] = 0.0
    for c in ("ausentes_home", "ausentes_away", "fuente_reporte"):
        if c not in g:
            g[c] = ""
    kickoff = pd.to_datetime(g["kickoff"], utc=True, errors="coerce")
    out = pd.DataFrame({
        "modelo": modelo,
        "origen": origen,
        "generado_en": generado_en,
        "season": g["season"].astype(int),
        "week": g["week"].astype(int),
        "game_id": g["game_id"],
        "kickoff": kickoff.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": g["home_team"],
        "away_team": g["away_team"],
        "p_home": g["p_home_pre"].round(4),
        "p_away": g["p_away_pre"].round(4),
        "favorito": g["favorito"],
        "p_fav": g["p_fav"].round(4),
        "elo_home_pre": g["elo_home_pre"],
        "elo_away_pre": g["elo_away_pre"],
        "ajuste_elo_home": g["ajuste_elo_home"],
        "ajuste_elo_away": g["ajuste_elo_away"],
        "ausentes_home": g["ausentes_home"].fillna(""),
        "ausentes_away": g["ausentes_away"].fillna(""),
        "fuente_reporte": g["fuente_reporte"].fillna(""),
        "params": params or "",
    })
    return out[COLUMNAS]


def registrar_pronosticos(games, modelo, semanas, season, origen="semanal",
                          generado_en=None, params=None, guardar=True):
    """
    Agrega al libro los pronósticos de `semanas` (int o lista) para `modelo`.
    - origen 'semanal'/'actualizacion': solo partidos que aún no empiezan.
    - origen 'backfill': partidos ya jugados; generado_en = 1 h antes del kickoff
      (la información usada es el reporte final de esa semana, previo al juego).
    Si la probabilidad de un partido no cambió respecto a su última versión,
    no se agrega otra (correr dos veces el mismo día no ensucia el libro).
    Regresa el DataFrame de renglones agregados.
    """
    if modelo not in MODELOS:
        raise ValueError(f"modelo debe ser uno de {list(MODELOS)}")
    semanas = [int(s) for s in np.atleast_1d(semanas)]
    ahora = generado_en or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    g = games[games["week"].isin(semanas)]
    nuevas = _filas(g, modelo, origen, ahora, params)

    kick = pd.to_datetime(nuevas["kickoff"], utc=True, errors="coerce")
    if origen == "backfill":
        nuevas["generado_en"] = (kick - timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        nuevas = nuevas[kick.notna()]
    else:
        gen = pd.Timestamp(ahora)
        nuevas = nuevas[kick.isna() | (kick > gen)]   # ya empezó -> no se registra

    libro = cargar_libro(season)
    if len(libro) and len(nuevas):
        ult = (libro[libro["modelo"] == modelo].sort_values("generado_en")
                     .drop_duplicates("game_id", keep="last").set_index("game_id")["p_home"])
        if origen == "backfill":
            nuevas = nuevas[~nuevas["game_id"].isin(ult.index)]   # backfill no pisa nada
        else:
            igual = nuevas["game_id"].map(ult).sub(nuevas["p_home"]).abs() < 5e-5
            nuevas = nuevas[~igual.fillna(False)]

    if guardar and len(nuevas):
        LIBRO_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat([libro, nuevas], ignore_index=True).to_csv(ruta_libro(season), index=False)
    return nuevas


def pronostico_vigente(libro):
    """Por partido y modelo: la última versión generada antes del kickoff."""
    if libro.empty:
        return libro
    l = libro.copy()
    gen = pd.to_datetime(l["generado_en"], utc=True)
    kick = pd.to_datetime(l["kickoff"], utc=True, errors="coerce")
    l = l[kick.isna() | (gen < kick)]
    return (l.sort_values("generado_en").drop_duplicates(["modelo", "game_id"], keep="last")
             .reset_index(drop=True))


def pronostico_publicado(libro, cambio_semana):
    """
    El historial del modelo que se publicó: Modelo 1 (elo_base) en las semanas
    < cambio_semana y Modelo 2 (elo_ausencias) desde cambio_semana. Toma lo
    congelado antes de cada kickoff (pronostico_vigente).
    """
    vig = pronostico_vigente(libro)
    if vig.empty:
        return vig
    usar = np.where(vig["week"] < cambio_semana, "elo_base", "elo_ausencias")
    return vig[vig["modelo"] == usar].reset_index(drop=True)


def evaluar_publicado(libro, games_ref, cambio_semana, threshold=0.5, hasta_semana=None):
    """evaluar_partidos() del historial publicado (ver pronostico_publicado)."""
    from src.weekly_content import evaluar_partidos

    pub = pronostico_publicado(libro, cambio_semana)
    if hasta_semana is not None and len(pub):
        pub = pub[pub["week"] < hasta_semana]
    if pub.empty:
        return pd.DataFrame()
    return evaluar_partidos(_como_games(pub, games_ref), threshold)


def _como_games(vig, games_ref):
    """
    Une los pronósticos congelados de un modelo con resultados reales y la
    línea de Las Vegas (de games_ref = enriquecer() del modelo base), en el
    formato que espera weekly_content.evaluar_partidos().
    """
    ref = games_ref[["game_id", "played", "home_score", "away_score", "p_home_vegas"]]
    g = vig.merge(ref, on="game_id", how="inner")
    g["p_home_pre"] = g["p_home"].astype(float)
    g["p_away_pre"] = g["p_away"].astype(float)
    g["fav_local"] = g["p_home_pre"] > 0.5
    g["favorito"] = np.where(g["fav_local"], g["home_team"], g["away_team"])
    g["p_fav"] = np.where(g["fav_local"], g["p_home_pre"], g["p_away_pre"])
    g["played"] = g["played"].astype(bool)
    return g


def evaluar_modelos(libro, games_ref, threshold=0.5, hasta_semana=None):
    """
    Métricas por semana y acumuladas de cada modelo con lo que se congeló.
    Solo se comparan partidos que tienen pronóstico de AMBOS modelos.
    Regresa (tabla_larga, evaluacion_por_partido).
    """
    from src.weekly_content import evaluar_partidos, metricas_por_semana

    vig = pronostico_vigente(libro)
    if vig.empty:
        return pd.DataFrame(), pd.DataFrame()
    if hasta_semana is not None:
        vig = vig[vig["week"] < hasta_semana]
    ambos = vig.groupby("game_id")["modelo"].nunique()
    vig = vig[vig["game_id"].isin(ambos[ambos == len(MODELOS)].index)]

    tablas, evs = [], []
    for modelo, v in vig.groupby("modelo"):
        ev = evaluar_partidos(_como_games(v, games_ref), threshold)
        if ev.empty:
            continue
        evs.append(ev.assign(modelo=modelo))
        tablas.append(metricas_por_semana(ev).assign(modelo=modelo))
    if not tablas:
        return pd.DataFrame(), pd.DataFrame()
    tabla = pd.concat(tablas, ignore_index=True)
    tabla.insert(0, "modelo", tabla.pop("modelo"))
    return tabla, pd.concat(evs, ignore_index=True)


def resumen_comparacion(tabla):
    """Una fila por modelo con lo acumulado a la última semana evaluada."""
    if tabla.empty:
        return tabla
    u = tabla[tabla["semana"] == tabla["semana"].max()]
    cols = {"modelo": "modelo", "semana": "hasta_semana", "acum_partidos": "partidos",
            "acum_record": "record", "acum_pct_aciertos": "pct_aciertos",
            "acum_brier": "brier", "acum_log_loss": "log_loss",
            "acum_vegas_brier": "vegas_brier", "acum_vegas_pct": "vegas_pct"}
    out = u[list(cols)].rename(columns=cols).reset_index(drop=True)
    out["nombre"] = out["modelo"].map(MODELOS)
    return out


def desacuerdos(ev):
    """Partidos donde los modelos eligieron distinto favorito y quién acertó."""
    if ev.empty:
        return ev
    ev = ev.assign(ganador=ev["ganador"].fillna("empate"))
    p = ev.pivot_table(index=["week", "game_id", "away_team", "home_team", "ganador"],
                       columns="modelo", values=["favorito", "p_fav"], aggfunc="first")
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    if not {"favorito_elo_base", "favorito_elo_ausencias"} <= set(p.columns):
        return pd.DataFrame()
    d = p[p["favorito_elo_base"] != p["favorito_elo_ausencias"]].copy()
    d["acerto"] = np.select(
        [d["ganador"] == d["favorito_elo_base"], d["ganador"] == d["favorito_elo_ausencias"]],
        ["Modelo 1", "Modelo 2"], default="empate")
    return d.sort_values(["week", "game_id"]).reset_index(drop=True)
