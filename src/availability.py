# src/availability.py
"""
Ausencias de titulares por equipo y partido (datos de nflreadpy).

Este módulo NO sabe nada de Elo: solo responde, para cada equipo en cada
partido, "¿cuántos titulares de cada grupo de posición le faltan?". El
modelo Elo + ausencias (src/elo_ausencias.py) convierte eso en puntos Elo.

Fuentes (nflreadpy)
-------------------
load_schedules       QB que inició cada partido (en semanas futuras, el QB
                     proyectado por nflverse).
load_injuries        Reporte oficial de lesiones de la semana (Out, Doubtful,
                     Questionable). Desde 2009. Sale miércoles-viernes.
load_rosters_weekly  Estatus de roster por semana (RES = reserva/IR, CUT,
                     TRD, ...). Atrapa a los titulares en IR, que ya no
                     aparecen en el reporte de lesiones. OJO: INA es la lista
                     de inactivos del día del partido; NO se usa porque el
                     martes todavía no existe.
load_snap_counts     % de snaps por jugador y partido. Define quién es
                     titular en RB, WR/TE, OL y defensa, y si de verdad jugó.
                     Desde 2012.
load_players         Puente de ids gsis (lesiones, rosters) <-> pfr (snaps).

Definiciones
------------
Titular QB     El QB designado de la temporada = el que inició la semana 1
               (misma regla del parche auditado). Un cambio permanente se
               registra en CAMBIOS_TITULAR.
Titular resto  Jugador con % de snaps >= UMBRAL_SNAPS[grupo] en al menos
               MIN_PARTIDOS_TITULAR de los últimos VENTANA_PARTIDOS partidos
               de su equipo ANTES de esa semana (nunca mira el futuro). En la
               semana 1 no hay historia de snaps: solo cuenta el QB. Un
               titular que lleva 4+ partidos fuera deja de serlo: el equipo
               ya tiene "nueva normalidad" y el Elo ya la absorbió.
Ausencia       A_pre  = probabilidad de que el titular NO juegue con la
                        información previa al partido (reporte de lesiones de
                        la semana, o de la anterior si aún no sale, + IR).
               A_real = 1 si de verdad no jugó (snaps = 0). Solo para
                        calibrar el efecto de una ausencia (backtest).
               Para el QB: si el QB que inicia (o el proyectado) no es el
               titular, la ausencia es 1.

Uso
---
    from src.availability import ausencias_temporada
    panel, info = ausencias_temporada(2026)
    panel[panel.week == 10]          # A_pre_QB, A_pre_RB, ... y 'ausentes'

Autoprueba:  python3 -m src.availability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nflreadpy as nfl

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros
# ══════════════════════════════════════════════════════════════════════════════
GRUPOS = ("QB", "RB", "WRTE", "OL", "DEF")

# Códigos viejos o de otras fuentes -> los del modelo (mismo criterio que
# src/elo.py:load_nfl_data, más los códigos gsis de rosters_weekly)
FRANCHISE_MAP = {
    "STL": "LA", "LAR": "LA", "SL": "LA",
    "OAK": "LV", "LVR": "LV",
    "SD": "LAC",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX",
}

POS_GRUPO = {
    "RB": "RB", "HB": "RB", "FB": "RB",
    "WR": "WRTE", "TE": "WRTE",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL",
    "DE": "DEF", "DT": "DEF", "NT": "DEF", "DL": "DEF",
    "LB": "DEF", "ILB": "DEF", "OLB": "DEF", "MLB": "DEF",
    "CB": "DEF", "S": "DEF", "FS": "DEF", "SS": "DEF", "DB": "DEF",
}
GRUPOS_OFENSIVOS = {"RB", "WRTE", "OL"}

UMBRAL_SNAPS = {"RB": 0.50, "WRTE": 0.60, "OL": 0.60, "DEF": 0.60}
VENTANA_PARTIDOS = 6
MIN_PARTIDOS_TITULAR = 3

# Estatus de rosters_weekly que significan "no está disponible para el equipo"
STATUS_FUERA_ROSTER = {"RES", "SUS", "RSN", "RSR", "CUT", "TRD", "TRC", "TRT",
                       "RET", "NWT", "DEV", "EXE", "PUP", "NON", "UFA", "UDF"}

SIN_DESIGNACION = "Sin designación"   # aparece en el reporte, sin estatus de juego

# P(el titular NO juega | estatus). "actual" = reporte de esa misma semana;
# "previa" = el de la semana anterior (cuando el de la semana aún no sale,
# p. ej. al correr el martes). Estimadas con estimar_prob_ausencia() sobre
# titulares 2012-2025; compare_models.py --recalibrar las vuelve a estimar y
# las guarda en data/processed/params_ausencias.csv (que tiene prioridad).
PROB_AUSENCIA = {   # estimado 2012-2025, ~118 mil titular-partido
    "actual": {"Out": 1.0, "Doubtful": 0.988, "Questionable": 0.267,
               "Probable": 0.015, SIN_DESIGNACION: 0.014},
    "previa": {"Out": 0.607, "Doubtful": 0.406, "Questionable": 0.138,
               "Probable": 0.052, SIN_DESIGNACION: 0.060},
}

# Cambios PERMANENTES de QB titular (el suplente es nombrado titular para el
# resto de la temporada). {temporada: {equipo: [(desde_semana, "Nombre"), ...]}}
# Ej.: {2026: {"MIN": [(10, "Kirk Cousins")]}}
CAMBIOS_TITULAR: dict[int, dict[str, list[tuple[int, str]]]] = {
    2026: {
        # Registrados en la semana 10: desde ahí, para no reescribir semanas ya
        # jugadas (lo congelado en el libro se generó sin saber del cambio).
        "ATL": [(10, "Michael Penix Jr.")],   # titular desde la semana 3
        "LV":  [(10, "Fernando Mendoza")],    # Cousins pasó a MIN
        "MIN": [(10, "Kirk Cousins")],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Carga de datos
# ══════════════════════════════════════════════════════════════════════════════
def _equipo(s):
    return s.replace(FRANCHISE_MAP)


def _grupo(pos):
    if not isinstance(pos, str) or not pos:
        return None
    return POS_GRUPO.get(pos.split("/")[0].strip())


def cargar_fuentes(seasons):
    """
    Descarga y normaliza las 5 fuentes para las temporadas pedidas.
    Regresa un dict de DataFrames de pandas: schedule, injuries, rosters,
    snaps, ids (pfr_id -> gsis_id).
    """
    seasons = [int(s) for s in np.atleast_1d(seasons)]
    print(f"Cargando datos de ausencias {min(seasons)}-{max(seasons)} (nflreadpy)...")

    sched = nfl.load_schedules(seasons=seasons).to_pandas()
    sched["home_team"] = _equipo(sched["home_team"])
    sched["away_team"] = _equipo(sched["away_team"])

    inj_seasons = [s for s in seasons if s >= 2009]
    inj = (nfl.load_injuries(inj_seasons).to_pandas() if inj_seasons
           else pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name",
                                      "position", "report_status"]))
    inj = inj[["season", "week", "team", "gsis_id", "full_name", "position", "report_status"]].copy()
    inj["team"] = _equipo(inj["team"])
    inj["report_status"] = inj["report_status"].fillna(SIN_DESIGNACION)
    inj = inj.dropna(subset=["gsis_id", "week"])
    inj[["season", "week"]] = inj[["season", "week"]].astype(int)
    # Un jugador puede aparecer dos veces (dos reportes en la semana): el más grave
    gravedad = {"Out": 4, "Doubtful": 3, "Questionable": 2, SIN_DESIGNACION: 1}
    inj["_g"] = inj["report_status"].map(gravedad).fillna(1)
    inj = (inj.sort_values("_g").drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
              .drop(columns="_g"))

    ros = nfl.load_rosters_weekly(seasons).to_pandas()
    ros = ros[["season", "week", "team", "gsis_id", "pfr_id", "full_name", "position", "status"]].copy()
    ros["team"] = _equipo(ros["team"])
    ros = ros.dropna(subset=["gsis_id", "week"])
    ros[["season", "week"]] = ros[["season", "week"]].astype(int)
    # Si un jugador sale dos veces en la semana (cambio de equipo), gana ACT
    ros["_act"] = (ros["status"] == "ACT").astype(int)
    ros = (ros.sort_values("_act").drop_duplicates(["season", "week", "gsis_id"], keep="last")
              .drop(columns="_act"))

    snap_seasons = [s for s in seasons if s >= 2012]
    snaps = (nfl.load_snap_counts(snap_seasons).to_pandas() if snap_seasons
             else pd.DataFrame(columns=["season", "week", "team", "pfr_player_id", "player",
                                        "position", "offense_pct", "defense_pct"]))
    snaps = snaps[["season", "week", "team", "pfr_player_id", "player", "position",
                   "offense_pct", "defense_pct"]].copy()
    snaps["team"] = _equipo(snaps["team"])
    snaps["grupo"] = snaps["position"].map(_grupo)
    snaps = snaps.dropna(subset=["grupo", "pfr_player_id"])
    snaps[["season", "week"]] = snaps[["season", "week"]].astype(int)
    snaps["pct"] = np.where(snaps["grupo"].isin(GRUPOS_OFENSIVOS),
                            snaps["offense_pct"], snaps["defense_pct"]).astype(float)

    players = nfl.load_players().to_pandas()[["pfr_id", "gsis_id"]].dropna()
    ids = pd.concat([players, ros[["pfr_id", "gsis_id"]].dropna()]).drop_duplicates("pfr_id")
    ids = dict(zip(ids["pfr_id"], ids["gsis_id"]))

    return {"schedule": sched, "injuries": inj, "rosters": ros, "snaps": snaps, "ids": ids}


# ══════════════════════════════════════════════════════════════════════════════
# Partidos por equipo y titulares
# ══════════════════════════════════════════════════════════════════════════════
def partidos_por_equipo(schedule):
    """Un renglón por (partido, equipo) con el QB que inició / proyectado."""
    s = schedule.dropna(subset=["week"])
    cols = ["game_id", "season", "week", "game_type"]
    played = s["home_score"].notna() & s["away_score"].notna()
    home = s[cols].assign(team=s["home_team"], opponent=s["away_team"], local=True,
                          qb_id=s["home_qb_id"], qb_name=s["home_qb_name"], played=played)
    away = s[cols].assign(team=s["away_team"], opponent=s["home_team"], local=False,
                          qb_id=s["away_qb_id"], qb_name=s["away_qb_name"], played=played)
    out = pd.concat([home, away], ignore_index=True)
    out[["season", "week"]] = out[["season", "week"]].astype(int)
    return out.sort_values(["season", "team", "week"]).reset_index(drop=True)


def qb_titulares(juegos, cambios=None):
    """
    QB titular designado para cada (season, team, week).
    Regla: el QB que inició el primer partido de temporada regular del
    equipo; desde la semana indicada en `cambios` se usa el nuevo titular.
    Regresa juegos con columnas qb1_id y qb1_name.
    """
    cambios = CAMBIOS_TITULAR if cambios is None else cambios
    reg = juegos[juegos["game_type"] == "REG"].sort_values("week")
    primero = reg.drop_duplicates(["season", "team"])[["season", "team", "qb_id", "qb_name"]]
    primero = primero.rename(columns={"qb_id": "qb1_id", "qb_name": "qb1_name"})
    out = juegos.merge(primero, on=["season", "team"], how="left")

    nombre_a_id = (juegos.dropna(subset=["qb_id"]).drop_duplicates("qb_name")
                         .set_index("qb_name")["qb_id"].to_dict())
    for season, equipos in (cambios or {}).items():
        for team, lista in equipos.items():
            for desde, nombre in sorted(lista):
                m = (out["season"] == season) & (out["team"] == team) & (out["week"] >= desde)
                out.loc[m, "qb1_name"] = nombre
                out.loc[m, "qb1_id"] = nombre_a_id.get(nombre, nombre)
    return out


def titulares_snaps(snaps, juegos):
    """
    Titulares no-QB de cada (season, team, week) a partir de los snaps de los
    partidos ANTERIORES de ese equipo en esa temporada.
    Regresa un DataFrame largo: season, week, team, grupo, pfr_id, nombre.
    """
    s = snaps.copy()
    s["es_tit"] = (s["pct"] >= s["grupo"].map(UMBRAL_SNAPS)).astype(int)
    # Grupo y nombre del jugador en ese equipo-temporada (el más frecuente)
    meta = (s.groupby(["season", "team", "pfr_player_id"])
              .agg(grupo=("grupo", lambda x: x.mode().iat[0]), nombre=("player", "last"))
              .reset_index())
    semanas = juegos.groupby(["season", "team"])["week"].apply(lambda x: sorted(set(x))).to_dict()

    filas = []
    for (season, team), g in s.groupby(["season", "team"]):
        piv = g.pivot_table(index="week", columns="pfr_player_id", values="es_tit",
                            aggfunc="max", fill_value=0).sort_index()
        for w in semanas.get((season, team), []):
            prev = piv[piv.index < w].tail(VENTANA_PARTIDOS)
            if prev.empty:
                continue
            req = min(MIN_PARTIDOS_TITULAR, len(prev))
            tit = prev.columns[prev.sum().to_numpy() >= req]
            filas.extend((season, w, team, p) for p in tit)
    tit = pd.DataFrame(filas, columns=["season", "week", "team", "pfr_player_id"])
    tit = tit.merge(meta, on=["season", "team", "pfr_player_id"], how="left")
    return tit.rename(columns={"pfr_player_id": "pfr_id"})


# ══════════════════════════════════════════════════════════════════════════════
# Probabilidad de ausencia
# ══════════════════════════════════════════════════════════════════════════════
def _estatus_reporte(largo, injuries):
    """
    Agrega a `largo` (season, week, team, gsis_id) el estatus del reporte de
    lesiones de esa semana (status_actual, si ya salió: hay_reporte) y el de
    la última semana anterior con reporte de ese equipo (status_previa,
    week_previa). _resolver_fuente() decide cuál usar.

    Un reporte cuenta solo si ya trae al menos una designación de juego (Out,
    Doubtful, Questionable...). A media semana nflverse publica reportes de
    práctica sin designaciones (p. ej. el miércoles solo los equipos del jueves):
    usarlos borraría lo que se sabía por el reporte anterior.
    """
    con_designacion = injuries["report_status"] != SIN_DESIGNACION
    rep = injuries.loc[con_designacion, ["season", "week", "team"]].drop_duplicates()
    rep_set = set(map(tuple, rep.to_numpy()))
    l = largo.copy()
    l["hay_reporte"] = [k in rep_set for k in zip(l["season"], l["week"], l["team"])]

    # Última semana con reporte < week (para la fuente 'previa')
    rep_s = rep.rename(columns={"week": "week_previa"}).sort_values("week_previa")
    l = l.sort_values("week")
    l["_wq"] = l["week"] - 1
    l = pd.merge_asof(l, rep_s, left_on="_wq", right_on="week_previa",
                      by=["season", "team"], direction="backward").drop(columns="_wq")

    est = injuries[["season", "week", "team", "gsis_id", "report_status"]]
    l = l.merge(est.rename(columns={"report_status": "status_actual"}),
                on=["season", "week", "team", "gsis_id"], how="left")
    l = l.merge(est.rename(columns={"report_status": "status_previa", "week": "week_previa"}),
                on=["season", "week_previa", "team", "gsis_id"], how="left")
    return l


def _resolver_fuente(l, reporte="auto"):
    """
    reporte='auto'   : el reporte de la semana si ya salió; si no, el anterior
                       (lo que se sabe el día que se corre).
    reporte='previa' : siempre el de la semana anterior (simula publicar el
                       martes, antes de que salga el reporte de la semana).
    """
    l = l.copy()
    usar_actual = l["hay_reporte"] if reporte == "auto" else pd.Series(False, index=l.index)
    l["report_status"] = np.where(usar_actual, l["status_actual"], l["status_previa"])
    l["week_fuente"] = np.where(usar_actual, l["week"], l["week_previa"])
    l["fuente"] = np.where(usar_actual, "actual",
                           np.where(l["week_previa"].notna(), "previa", "sin reporte"))
    return l


def _estatus_roster(largo, rosters):
    """Estatus de roster en esa semana (o la última disponible si aún no hay)."""
    r = rosters[["season", "week", "gsis_id", "team", "status"]].rename(
        columns={"team": "team_roster", "status": "status_roster", "week": "week_roster"})
    r = r.sort_values("week_roster")
    l = largo.sort_values("week")
    l = pd.merge_asof(l, r, left_on="week", right_on="week_roster",
                      by=["season", "gsis_id"], direction="backward")
    fuera = (l["status_roster"].isin(STATUS_FUERA_ROSTER)
             | (l["team_roster"].notna() & (l["team_roster"] != l["team"])))
    l["fuera_roster"] = fuera
    return l


def _prob(row_fuente, row_status, probs):
    if not isinstance(row_status, str):
        return 0.0
    return float(probs.get(row_fuente, {}).get(row_status, 0.0))


def _aplicar_prob(l, probs):
    p = np.array([_prob(f, s, probs) for f, s in zip(l["fuente"], l["report_status"])])
    p = np.where(l["fuera_roster"], 1.0, p)
    return p


def _etiqueta(r):
    if r.get("qb_distinto", False):
        return f"{r['nombre']} (QB, inicia {r['qb_name']})"
    if r["fuera_roster"]:
        est = "IR" if r["status_roster"] == "RES" else (
            "otro equipo" if r["team_roster"] != r["team"] else r["status_roster"])
    else:
        est = r["report_status"]
        if r["fuente"] == "previa":
            est = f"{est} sem. {int(r['week_fuente'])}"
    return f"{r['nombre']} ({r['grupo']}, {est})"


# ══════════════════════════════════════════════════════════════════════════════
# Panel de ausencias
# ══════════════════════════════════════════════════════════════════════════════
def construir_panel(fuentes, probs=None, cambios=None, reporte="auto", detalle=False):
    """
    Un renglón por (partido, equipo) con:
        A_pre_<grupo>   titulares ausentes esperados con info previa al partido
        A_real_<grupo>  titulares que de verdad no jugaron (solo jugados)
        n_tit_<grupo>   titulares identificados
        ausentes        texto con los ausentes (p >= 0.25) para revisar a mano
        qb1_name, qb_name, fuente_reporte
    reporte: 'auto' o 'previa' (ver _resolver_fuente).
    Con detalle=True regresa también la tabla larga jugador-partido.
    """
    largo, juegos = tabla_larga(fuentes, cambios)
    panel, largo = agregar_panel(largo, juegos, probs, reporte)
    return (panel, largo) if detalle else panel


def tabla_larga(fuentes, cambios=None):
    """Tabla jugador-partido (titulares) con estatus de reporte y de roster."""
    juegos = qb_titulares(partidos_por_equipo(fuentes["schedule"]), cambios)

    # ── QB: un renglón por equipo-partido ─────────────────────────────────────
    qb = juegos[["season", "week", "team", "game_id", "played", "qb_id", "qb_name",
                 "qb1_id", "qb1_name"]].copy()
    qb["grupo"] = "QB"
    qb["gsis_id"] = qb["qb1_id"]
    qb["nombre"] = qb["qb1_name"]
    qb["qb_distinto"] = qb["qb_id"].notna() & qb["qb1_id"].notna() & (qb["qb_id"] != qb["qb1_id"])
    qb["no_jugo"] = np.where(qb["played"], qb["qb_distinto"].astype(float), np.nan)

    # ── Resto: titulares por snaps ───────────────────────────────────────────
    tit = titulares_snaps(fuentes["snaps"], juegos)
    tit["gsis_id"] = tit["pfr_id"].map(fuentes["ids"])
    tit = tit.merge(juegos[["season", "week", "team", "game_id", "played"]],
                    on=["season", "week", "team"], how="inner")
    jugo = fuentes["snaps"][fuentes["snaps"]["pct"] > 0][["season", "week", "team", "pfr_player_id"]]
    jugo = set(map(tuple, jugo.to_numpy()))
    tit["no_jugo"] = [
        (0.0 if (s, w, t, p) in jugo else 1.0) if pl else np.nan
        for s, w, t, p, pl in zip(tit["season"], tit["week"], tit["team"], tit["pfr_id"], tit["played"])
    ]
    # Snaps de un partido jugado que aún no se publican -> no hay A_real
    con_snaps = set(map(tuple, fuentes["snaps"][["season", "week", "team"]].drop_duplicates().to_numpy()))
    sin = [(s, w, t) not in con_snaps for s, w, t in zip(tit["season"], tit["week"], tit["team"])]
    tit.loc[sin, "no_jugo"] = np.nan
    tit["qb_distinto"] = False

    largo = pd.concat([qb, tit], ignore_index=True, sort=False)
    largo = _estatus_reporte(largo, fuentes["injuries"])
    largo = _estatus_roster(largo, fuentes["rosters"])
    return largo, juegos


def agregar_panel(largo, juegos, probs=None, reporte="auto"):
    """Aplica P(ausencia | estatus) y suma por equipo-partido y grupo."""
    probs = probs or PROB_AUSENCIA
    largo = _resolver_fuente(largo, reporte)
    es_qb = (largo["grupo"] == "QB").to_numpy()
    # Tras un traspaso, rosters_weekly puede seguir con el jugador en su equipo
    # anterior unos días. Si el calendario ya lo lista como el QB que inicia por
    # ESTE equipo, se le cree al calendario: solo cuenta el estatus (IR, etc.).
    listado = es_qb & largo["qb_id"].notna().to_numpy() & ~largo["qb_distinto"].to_numpy(bool)
    largo.loc[listado, "fuera_roster"] = largo.loc[listado, "status_roster"].isin(STATUS_FUERA_ROSTER)
    p = _aplicar_prob(largo, probs)
    # QB: si ya se sabe quién inicia y no es el titular -> 1. Si el partido ya
    # se jugó, el QB que inició es conocido antes del kickoff -> 0/1 exacto.
    p_qb = np.where(largo["qb_distinto"], 1.0,
                    np.where(largo["played"], 0.0, p))
    largo["p_ausente"] = np.where(es_qb, p_qb, p)
    largo.loc[largo["gsis_id"].isna() & ~es_qb, "p_ausente"] = 0.0

    agg = {}
    for g in GRUPOS:
        m = largo["grupo"] == g
        agg[f"A_pre_{g}"] = largo["p_ausente"].where(m, 0.0)
        agg[f"A_real_{g}"] = largo["no_jugo"].where(m, 0.0)
        agg[f"n_tit_{g}"] = m.astype(int)
    largo = largo.assign(**agg)
    llaves = ["season", "week", "team", "game_id"]
    panel = largo.groupby(llaves).agg(
        **{c: (c, "sum") for c in agg if not c.startswith("A_real")},
        **{c: (c, lambda x: x.sum(min_count=1)) for c in agg if c.startswith("A_real")},
    ).reset_index()

    vis = largo[largo["p_ausente"] >= 0.25].copy()
    vis["_txt"] = [_etiqueta(r) for _, r in vis.iterrows()]
    vis = vis.sort_values(["grupo"], key=lambda x: x.map({g: i for i, g in enumerate(GRUPOS)}))
    txt = vis.groupby(llaves)["_txt"].apply(lambda x: "; ".join(x)).rename("ausentes")
    fuente = (largo.groupby(llaves)
                   .apply(lambda d: _fuente_texto(d))
                   .rename("fuente_reporte"))
    panel = (panel.merge(txt, on=llaves, how="left").merge(fuente, on=llaves, how="left")
                  .merge(juegos[llaves + ["qb1_name", "qb_name", "played"]], on=llaves, how="left"))
    panel["ausentes"] = panel["ausentes"].fillna("")
    panel = panel.sort_values(["season", "week", "team"]).reset_index(drop=True)
    return panel, largo


def _fuente_texto(d):
    f = d["fuente"].iloc[0]
    if f == "actual":
        return f"reporte sem. {int(d['week'].iloc[0])}"
    if f == "previa":
        return f"provisional: reporte sem. {int(d['week_fuente'].iloc[0])}"
    return "sin reporte"


# ══════════════════════════════════════════════════════════════════════════════
# Estimaciones y utilidades
# ══════════════════════════════════════════════════════════════════════════════
def estimar_prob_ausencia(largo, temporadas=None):
    """
    P(el titular no juega | estatus del reporte), con dos fuentes:
        actual  reporte de esa misma semana (lo que se sabe el viernes)
        previa  reporte de la semana anterior (lo que se sabe el martes)
    Solo titulares no-QB que siguen en el roster, de partidos jugados con
    snaps publicados. `largo` = primer valor de tabla_larga().
    """
    base = largo[(largo["grupo"] != "QB") & largo["no_jugo"].notna() & ~largo["fuera_roster"]]
    if temporadas is not None:
        base = base[base["season"].isin(temporadas)]
    tablas = []
    for fuente, col, filtro in (("actual", "status_actual", base["hay_reporte"]),
                                ("previa", "status_previa", base["week_previa"].notna())):
        l = base[filtro]
        est = l[col].fillna("(no aparece)")
        t = l.groupby(est)["no_jugo"].agg(["size", "mean"]).reset_index()
        t.columns = ["estatus", "n", "p_no_juega"]
        tablas.append(t.assign(fuente=fuente))
    return pd.concat(tablas, ignore_index=True)[["fuente", "estatus", "n", "p_no_juega"]]


def tabla_a_probs(tabla):
    """Tabla de estimar_prob_* -> dict como PROB_AUSENCIA (sin '(no aparece)')."""
    probs = {"actual": {}, "previa": {}}
    for r in tabla.itertuples(index=False):
        if r.estatus != "(no aparece)":
            probs[r.fuente][r.estatus] = round(float(r.p_no_juega), 3)
    return probs


def sugerir_cambios_qb(panel, season, n=3):
    """
    Equipos cuyo QB titular no ha iniciado los últimos `n` partidos jugados.
    Si el cambio es permanente (no una lesión), conviene registrarlo en
    CAMBIOS_TITULAR para dejar de penalizar al equipo.
    """
    t = panel[panel["season"] == season].sort_values("week")
    titular_hoy = t.groupby("team")["qb1_name"].last()   # ya con CAMBIOS_TITULAR aplicado
    p = t[t["played"]]
    out = []
    for team, d in p.groupby("team"):
        ult = d.tail(n)
        if len(ult) == n and (ult["qb_name"] != ult["qb1_name"]).all() \
                and ult["qb_name"].nunique() == 1 \
                and titular_hoy.get(team) != ult["qb_name"].iat[-1]:
            out.append({"team": team, "qb_titular": ult["qb1_name"].iat[-1],
                        "inicia_desde_hace": n, "qb_actual": ult["qb_name"].iat[-1]})
    return pd.DataFrame(out)


def ausencias_temporada(season, probs=None, cambios=None):
    """
    Panel de ausencias de una temporada (lo que usa el flujo semanal) y un
    dict con información del último reporte disponible.
    """
    fuentes = cargar_fuentes([season])
    panel = construir_panel(fuentes, probs=probs, cambios=cambios)
    inj = fuentes["injuries"]
    ros = fuentes["rosters"]
    info = {
        "ultima_semana_reporte": int(inj["week"].max()) if len(inj) else None,
        "ultima_semana_roster": int(ros["week"].max()) if len(ros) else None,
        "ultima_semana_snaps": int(fuentes["snaps"]["week"].max()) if len(fuentes["snaps"]) else None,
        "sugerencias_qb": sugerir_cambios_qb(panel, season),
    }
    return panel, info


# ══════════════════════════════════════════════════════════════════════════════
# Autoprueba
# ══════════════════════════════════════════════════════════════════════════════
QB1_2026_PARCHE = {
    "ARI": "Jacoby Brissett", 
    "ATL": "Cooper Rush", 
    "BAL": "Lamar Jackson",
    "BUF": "Josh Allen", 
    "CAR": "Bryce Young", 
    "CHI": "Caleb Williams",
    "CIN": "Joe Burrow", 
    "CLE": "Deshaun Watson", 
    "DAL": "Dak Prescott",
    "DEN": "Bo Nix", 
    "DET": "Jared Goff", 
    "GB": "Jordan Love", 
    "HOU": "C.J. Stroud",
    "IND": "Daniel Jones", 
    "JAX": "Trevor Lawrence", 
    "KC": "Patrick Mahomes",
    "LA": "Matthew Stafford", 
    "LAC": "Justin Herbert", 
    "LV": "Kirk Cousins",
    "MIA": "Malik Willis", 
    "MIN": "Kyler Murray", 
    "NE": "Drake Maye",
    "NO": "Tyler Shough", 
    "NYG": "Jaxson Dart", 
    "NYJ": "Geno Smith",
    "PHI": "Jalen Hurts", 
    "PIT": "Aaron Rodgers", 
    "SEA": "Sam Darnold",
    "SF": "Brock Purdy", 
    "TB": "Baker Mayfield", 
    "TEN": "Cam Ward",
    "WAS": "Jayden Daniels",
}

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)
    panel, info = ausencias_temporada(2026, cambios={})
    qb1 = panel[panel["week"] == panel.groupby("team")["week"].transform("min")]
    derivado = dict(zip(qb1["team"], qb1["qb1_name"]))
    assert derivado == QB1_2026_PARCHE, {k: (v, QB1_2026_PARCHE.get(k)) for k, v in derivado.items()
                                         if v != QB1_2026_PARCHE.get(k)}
    print("OK: QB1_2026 derivado de load_schedules == diccionario del parche (32/32)")
    print(info)
    s9 = panel[panel["week"] == 9]
    print(s9[["team", "A_pre_QB", "A_pre_RB", "A_pre_WRTE", "A_pre_OL", "A_pre_DEF",
              "fuente_reporte", "ausentes"]].to_string(index=False))
    for team in ("BAL", "SEA", "MIN"):
        assert s9.loc[s9["team"] == team, "A_pre_QB"].iat[0] == 1.0, team
    print("OK: BAL, SEA y MIN aparecen sin su QB titular en la semana 9")
