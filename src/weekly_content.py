#!/usr/bin/env python3
# src/weekly_content.py
"""
Contenido semanal · "Pronosticando la NFL con un modelo de ELO"
================================================================

Córrelo cada martes (después del Monday Night Football). En un solo paso:

  1. DATOS     Descarga el calendario con nflreadpy, recalcula el ELO partido a
               partido con los resultados reales y actualiza los CSV de
               data/processed/ (los mismos archivos que genera el Notebook 05).
  2. MÉTRICAS  Evalúa al modelo con los partidos ya jugados. El umbral de
               probabilidad para contar un pick es un parámetro (--threshold,
               default: mayor a 0.50).
  3. GRÁFICAS  Las mismas imágenes en dos tamaños, con el mismo lienzo de papel
               cuadriculado (ver FORMATOS):
                 reels/semana_XX/size_1/  1080x902   (publicación)
                 reels/semana_XX/size_2/  1080x1920  (reel / historia)
               Cada carpeta lleva:
                 01  aciertos por semana + acumulado
                 02  quién sube / quién cae en ELO (la semana pasada)
                 03  quién sube / quién cae en ELO (desde la pretemporada)
                 04  top 5 según el ELO
                 XX  picks de la semana, en orden de juego y con escudos
                     (XX = semana - 1: semana 6 -> 05_picks_semana_06)
                 06  el más seguro
                 07  la moneda al aire
                 08  la sorpresa
                 09  evolución del ELO de la temporada (top 5 en color)
                 10  evolución del ELO por división (los 32 equipos)
                 partidos/  una historia por partido (récord, escudos,
                            probabilidad y comentario) + comentarios.md
               + resumen.md con los números para el guion y las tablas en CSV
               (en la raíz de reels/semana_XX/).

Uso (desde la raíz del repo):
    python src/weekly_content.py                    # semana automática
    python src/weekly_content.py --formatos size_1  # solo las imágenes 1080x902
    python src/weekly_content.py --threshold 0.6    # solo cuenta picks con > 60 %
    python src/weekly_content.py --semana 5         # rehacer el contenido de la semana 5
    python src/weekly_content.py --sin-actualizar   # no reescribe data/processed

Desde un notebook:
    import sys; sys.path.append('..')
    from src.weekly_content import generar_contenido
    res = generar_contenido(threshold=0.5)
    res["metricas"]            # tabla semana a semana
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nflreadpy as nfl  # noqa: E402

from src.weekly_probabilities import (  # noqa: E402
    build_weekly_snapshots,
    current_week_probabilities,
)

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros
# ══════════════════════════════════════════════════════════════════════════════
# ELO (calibrados en el Notebook 02 — mismos que los Notebooks 04 y 05)
HOME_ADVANTAGE = 42.2
K_REGULAR = 40
K_PLAYOFFS = 48
REGRESSION_FACTOR = 0.40
GLOBAL_MEAN = 1503.1

SEASON = 2026
THRESHOLD = 0.50        # un partido cuenta como pick si prob. del favorito > THRESHOLD
ALTA_CONFIANZA = 0.60   # desde aquí la barra del pick va en color acento
ZONA_HORARIA = "America/Mexico_City"
TOP_N = 5
MOSTRAR_VEGAS = True    # comparar contra el favorito de Las Vegas en la gráfica 01

DATA_DIR = ROOT / "data" / "processed"
LOGOS_DIR = ROOT / "data" / "logos"      # caché local de escudos
REELS_DIR = ROOT / "reels"               # una carpeta por semana

# ══════════════════════════════════════════════════════════════════════════════
# Estilo — papel cuadriculado del reel
# ══════════════════════════════════════════════════════════════════════════════
FONDO = "#EFE6DC"
TINTA = "#1E1B18"
ACENTO = "#7A6A50"       # dorado/café: favorito, lo que sube
FRIO = "#C9CFE0"         # azul grisáceo: rival, lo secundario
ROJO = "#B5523B"         # lo que cae
GRIS = "#8C8578"
CUADRICULA = "#D9CEC0"
PISTA = "#E2D8CB"        # fondo de las barras de probabilidad
FRIO_TEXTO = "#8E97B5"   # FRIO con contraste suficiente para texto

W, H, DPI = 9, 16, 120   # 1080 x 1920 — el lienzo de diseño (ver FORMATOS)
# Zona segura: Instagram/TikTok tapan arriba (~12 %) y abajo (~18 %)
SAFE_TOP, SAFE_BOTTOM = 0.84, 0.20

ESTILO = {
    "font.family": "DejaVu Sans",
    "text.color": TINTA,
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA,
    "ytick.color": TINTA,
}

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]
ZONAS_CORTAS = {"America/Mexico_City": "CDMX", "America/New_York": "ET",
                "America/Chicago": "CT", "America/Los_Angeles": "PT",
                "Europe/Madrid": "Madrid"}


DECIMALES = 1   # decimales en las probabilidades que se muestran


def fmt_pct(p, d=DECIMALES):
    """0.5012 -> '50.1%'"""
    return f"{p:.{d}%}"


def fmt_par(p_a, p_b, d=DECIMALES):
    """
    Formatea dos probabilidades complementarias para que siempre sumen 100.
    Redondea la primera y la segunda es 100 menos la primera, para evitar
    casos como 50.1% vs 50.0%.
    """
    a = round(p_a * 100, d)
    return f"{a:.{d}f}%", f"{100 - a:.{d}f}%"
# ══════════════════════════════════════════════════════════════════════════════
# 1 · Datos — lo mismo que hace el Notebook 05
# ══════════════════════════════════════════════════════════════════════════════
def cargar_datos(season=SEASON):
    """Calendario con marcadores reales (nflreadpy) + ELO de cierre del año previo."""
    schedule = nfl.load_schedules(seasons=[season]).to_pandas()
    elo_prev = pd.read_csv(DATA_DIR / f"elo_final_{season - 1}.csv")
    final_elos = dict(zip(elo_prev["team"], elo_prev["elo_final"]))

    n_played = int(schedule[["home_score", "away_score"]].notna().all(axis=1).sum())
    print(f"Partidos {season}      : {len(schedule)}")
    print(f"Ya jugados          : {n_played}")
    print(f"Pendientes          : {len(schedule) - n_played}")
    print(f"Equipos con ELO base: {len(final_elos)}")
    return schedule, final_elos


def calcular_elo(schedule, final_elos):
    """games_df (ELO pre/post y probabilidad por partido) y elo_evolution_df."""
    return build_weekly_snapshots(
        schedule_df=schedule,
        final_elos_prev_season=final_elos,
        home_advantage=HOME_ADVANTAGE,
        k_regular=K_REGULAR,
        k_playoffs=K_PLAYOFFS,
        regression_factor=REGRESSION_FACTOR,
        global_mean=GLOBAL_MEAN,
    )


def guardar_archivos(games_df, elo_evolution_df, season=SEASON, data_dir=DATA_DIR):
    """Mismos archivos que la §4 del Notebook 05."""
    weekly_dir = data_dir / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(data_dir / f"weekly_predictions_{season}.csv", index=False)
    elo_evolution_df.to_csv(data_dir / f"elo_evolution_{season}.csv", index=False)
    for week, week_df in games_df.groupby("week"):
        week_df.to_csv(weekly_dir / f"week_{int(week):02d}_probabilities.csv", index=False)
    print(f"Guardado: weekly_predictions_{season}.csv, elo_evolution_{season}.csv "
          f"y {games_df['week'].nunique()} archivos en data/processed/weekly/")


def _prob_moneyline(ml):
    """Moneyline americano -> probabilidad implícita (con margen de la casa)."""
    ml = pd.to_numeric(ml, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def enriquecer(games_df, schedule, zona=ZONA_HORARIA):
    """
    Agrega a games_df lo que el motor de ELO no guarda: horario (en la zona
    elegida), probabilidad implícita de Las Vegas y el favorito del modelo.
    """
    cols = ["season", "week", "home_team", "away_team", "game_id",
            "gameday", "gametime", "home_moneyline", "away_moneyline"]
    info = schedule[cols].copy()
    info["season"] = info["season"].astype(int)
    info["week"] = info["week"].astype(int)

    g = games_df.copy()
    g["season"] = g["season"].astype(int)
    g["week"] = g["week"].astype(int)
    g = g.merge(info, on=["season", "week", "home_team", "away_team"],
                how="left", validate="one_to_one")

    # nflverse publica gameday + gametime en hora del Este (ET)
    et, local = ZoneInfo("America/New_York"), ZoneInfo(zona)

    def _kickoff(gameday, gametime):
        if pd.isna(gameday):
            return pd.NaT
        hora = gametime if isinstance(gametime, str) and ":" in gametime else "13:00"
        dt = datetime.strptime(f"{str(gameday)[:10]} {hora[:5]}", "%Y-%m-%d %H:%M")
        return pd.Timestamp(dt.replace(tzinfo=et).astimezone(local))

    g["kickoff"] = [_kickoff(d, t) for d, t in zip(g["gameday"], g["gametime"])]

    # Las Vegas: prob. implícita del moneyline, quitando el margen de la casa
    ph, pa = _prob_moneyline(g["home_moneyline"]), _prob_moneyline(g["away_moneyline"])
    g["p_home_vegas"] = ph / (ph + pa)

    # Favorito del modelo
    g["fav_local"] = g["p_home_pre"] > 0.5
    g["favorito"] = np.where(g["fav_local"], g["home_team"], g["away_team"])
    g["rival"] = np.where(g["fav_local"], g["away_team"], g["home_team"])
    g["p_fav"] = np.where(g["fav_local"], g["p_home_pre"], g["p_away_pre"])
    return g


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Métricas de evaluación
# ══════════════════════════════════════════════════════════════════════════════
def evaluar_partidos(games, threshold=THRESHOLD):
    """
    Una fila por partido jugado (empates incluidos) con el resultado del pick.

    Definiciones
    ------------
    favorito  : el equipo con prob. > 50 % según el modelo.
    es_pick   : True si la prob. del favorito es > threshold. Con 0.50 todos los
                partidos son pick; con 0.60 solo los de confianza alta.
    empate    : el partido terminó empatado (ni acierto ni fallo).
    acierto   : el favorito del modelo ganó.
    fallo     : el favorito del modelo perdió.
    gano_local: 1 ganó el local · 0 ganó la visita · 0.5 empate (convención
                FiveThirtyEight para Brier / log loss).
    p_ganador : la prob. que el modelo le daba al que terminó ganando
                (la más baja de la semana = el batacazo). NaN en empates.
    vegas_fav : favorito del moneyline de cierre (nflverse), si hay línea.
    """
    j = games[games["played"]].copy()
    local_gana = j["home_score"] > j["away_score"]
    j["empate"] = j["home_score"] == j["away_score"]
    j["gano_local"] = np.where(j["empate"], 0.5, local_gana.astype(float))
    j["ganador"] = pd.Series(np.where(local_gana, j["home_team"], j["away_team"]),
                             index=j.index).mask(j["empate"])
    j["es_pick"] = j["p_fav"] > threshold
    j["acierto"] = (j["favorito"] == j["ganador"]) & ~j["empate"]
    j["fallo"] = ~j["acierto"] & ~j["empate"]
    j["p_ganador"] = pd.Series(np.where(local_gana, j["p_home_pre"], j["p_away_pre"]),
                               index=j.index).mask(j["empate"])

    sin_linea = j["p_home_vegas"].isna() | (j["p_home_vegas"] == 0.5)
    vegas_fav = np.where(j["p_home_vegas"] > 0.5, j["home_team"], j["away_team"])
    j["vegas_fav"] = pd.Series(vegas_fav, index=j.index).mask(sin_linea)
    return j


def record_af(aciertos, total, empates=0):
    """Récord estilo NFL aplicado al modelo: aciertos-fallos(-empates).
    `total` incluye los empates; el tercer número solo aparece si hay alguno."""
    aciertos, total, empates = int(aciertos), int(total), int(empates)
    rec = f"{aciertos}-{total - aciertos - empates}"
    return f"{rec}-{empates}" if empates else rec


def pct_nfl(aciertos, total, empates=0):
    """% de aciertos como el win% de la NFL: un empate cuenta como medio acierto."""
    return (aciertos + 0.5 * empates) / total if total else float("nan")


def _metricas(ev):
    """
    Métricas de un conjunto de partidos evaluados.

    partidos            jugados, empates incluidos
    picks               partidos con prob. del favorito > threshold
    cobertura           picks / partidos
    aciertos, errores   de los picks, cuántos ganó / perdió el favorito
    empates             picks que terminaron empatados
    record              aciertos-fallos(-empates), como el récord de un equipo
                        (19-13 o 19-12-1)
    pct_aciertos        (aciertos + 0.5·empates) / picks, como el win% de la NFL
                        -> "de los que dijo que ganaban, ¿cuántos ganaron?"
    aciertos_esperados  suma de las probs. de los picks. Si el modelo está bien
                        calibrado, aciertos ≈ aciertos_esperados.
    brier               promedio de (p_local - ganó_local)²  (moneda = 0.25)
    brier_skill         1 - brier / 0.25  (> 0 = mejor que la moneda)
    log_loss            pérdida logarítmica (moneda = 0.693)
    *_local / *_visita  picks y aciertos cuando eligió al local / al visitante
    siempre_local_*     estrategia ingenua "siempre el local" en esos mismos picks
    vegas_*             favorito de Las Vegas en esos mismos picks (si hay línea)
    acuerdo_vegas       % de picks donde el modelo y Las Vegas eligen igual
    """
    nan = float("nan")
    n = len(ev)
    picks = ev[ev["es_pick"]]
    n_p = len(picks)
    ac = int(picks["acierto"].sum())
    em = int(picks["empate"].sum())

    y = ev["gano_local"].to_numpy(dtype=float)
    p = np.clip(ev["p_home_pre"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2)) if n else nan
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if n else nan

    loc, vis = picks[picks["fav_local"]], picks[~picks["fav_local"]]
    con_v = picks[picks["vegas_fav"].notna()]
    v_ac = int((con_v["vegas_fav"] == con_v["ganador"]).sum())
    v_em = int(con_v["empate"].sum())
    todos_v = ev[ev["p_home_vegas"].notna()]
    vegas_brier = (float(np.mean((todos_v["p_home_vegas"] - todos_v["gano_local"]) ** 2))
                   if len(todos_v) else nan)

    return {
        "partidos": n,
        "picks": n_p,
        "cobertura": n_p / n if n else nan,
        "aciertos": ac,
        "errores": n_p - ac - em,
        "empates": em,
        "record": record_af(ac, n_p, em),
        "pct_aciertos": pct_nfl(ac, n_p, em),
        "aciertos_esperados": float(picks["p_fav"].sum()),
        "brier": brier,
        "brier_skill": 1 - brier / 0.25 if n else nan,
        "log_loss": log_loss,
        "picks_local": len(loc),
        "aciertos_local": int(loc["acierto"].sum()),
        "empates_local": int(loc["empate"].sum()),
        "picks_visita": len(vis),
        "aciertos_visita": int(vis["acierto"].sum()),
        "empates_visita": int(vis["empate"].sum()),
        "siempre_local_aciertos": int((picks["gano_local"] == 1).sum()),
        "siempre_local_pct": picks["gano_local"].mean() if n_p else nan,
        "vegas_partidos": len(con_v),
        "vegas_aciertos": v_ac,
        "vegas_empates": v_em,
        "vegas_pct": pct_nfl(v_ac, len(con_v), v_em),
        "vegas_brier": vegas_brier,
        "acuerdo_vegas": (float((con_v["vegas_fav"] == con_v["favorito"]).mean())
                          if len(con_v) else nan),
    }


def metricas_por_semana(ev):
    """Métricas de cada semana y acumuladas hasta esa semana (prefijo acum_)."""
    filas = []
    for w in sorted(ev["week"].unique()):
        fila = {"semana": int(w)}
        fila.update(_metricas(ev[ev["week"] == w]))
        fila.update({f"acum_{k}": v for k, v in _metricas(ev[ev["week"] <= w]).items()})
        filas.append(fila)
    return pd.DataFrame(filas)


def calibracion(ev):
    """¿Cuando el modelo dice 70 %, gana el favorito ~70 % de las veces?"""
    bins = [0.5, 0.6, 0.7, 0.8, 1.0 + 1e-9]
    etiquetas = ["50–60%", "60–70%", "70–80%", "80%+"]
    t = ev.assign(rango=pd.cut(ev["p_fav"], bins=bins, labels=etiquetas, right=False))
    out = (t.groupby("rango", observed=False)
             .agg(partidos=("acierto", "size"), aciertos=("acierto", "sum"),
                  empates=("empate", "sum"), prob_media=("p_fav", "mean"))
             .reset_index())
    out["pct_real"] = ((out["aciertos"] + 0.5 * out["empates"])
                       / out["partidos"].replace(0, np.nan))
    out["record"] = [record_af(a, n, e) for a, n, e
                     in zip(out["aciertos"], out["partidos"], out["empates"])]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Selección de datos para las gráficas
# ══════════════════════════════════════════════════════════════════════════════
def tabla_elo(elo_evolution_df):
    """Equipo x semana con el ELO al inicio de cada semana."""
    return elo_evolution_df.pivot(index="team", columns="week", values="elo_pre_week")


def resultados_semana(games, semana):
    """{equipo: 'Ganó 27–7 vs LA'} para los partidos jugados de esa semana."""
    ctx = {}
    g = games[(games["week"] == semana) & games["played"]]
    for r in g.itertuples():
        hs, as_ = int(r.home_score), int(r.away_score)
        for team, pf, pc, rival, local in ((r.home_team, hs, as_, r.away_team, True),
                                           (r.away_team, as_, hs, r.home_team, False)):
            res = "Ganó" if pf > pc else ("Perdió" if pf < pc else "Empató")
            ctx[team] = f"{res} {pf}–{pc} {'vs' if local else 'en'} {rival}"
    return ctx


def records_equipos(games, semana):
    """{equipo: (ganados, perdidos, empates)} con los partidos jugados antes de `semana`."""
    g = games[(games["week"] < semana) & games["played"]]
    rec = {t: (0, 0, 0) for t in pd.concat([games["home_team"], games["away_team"]]).unique()}
    for r in g.itertuples():
        for team, pf, pc in ((r.home_team, r.home_score, r.away_score),
                             (r.away_team, r.away_score, r.home_score)):
            w, l, t = rec[team]
            rec[team] = (w + int(pf > pc), l + int(pf < pc), t + int(pf == pc))
    return rec


def fmt_record(rec):
    """(2, 1, 0) -> '2-1' · (2, 1, 1) -> '2-1-1'."""
    w, l, t = rec
    return f"{w}-{l}" + (f"-{t}" if t else "")


def records(games, semana):
    """{equipo: 'Récord 2-1'} — texto para la gráfica de ELO de la temporada."""
    return {k: f"Récord {fmt_record(v)}" for k, v in records_equipos(games, semana).items()}


def _pct_victorias(rec):
    w, l, t = rec
    n = w + l + t
    return (w + 0.5 * t) / n if n else 0.5


def comentario_partido(r, rec, rank):
    """
    Comentario corto para la historia de cada partido, en tres frases:
      1. Probabilidad  — qué tan favorito es el pick del modelo.
      2. Elo / localía — la brecha de Elo, o si la localía es lo que decide.
      3. Récord        — invictos, sin victorias o si el récord contradice al modelo.
    """
    fav, riv, p = r.favorito, r.rival, float(r.p_fav)
    elo_f = float(r.elo_home_pre if r.fav_local else r.elo_away_pre)
    elo_r = float(r.elo_away_pre if r.fav_local else r.elo_home_pre)
    rk_f, rk_r = rank.get(fav, "–"), rank.get(riv, "–")

    # 1 · Probabilidad (mismo valor redondeado que se ve en la imagen)
    pct_num = round(p * 100, DECIMALES)   # 57.3 → para comparar contra los cortes
    pct = fmt_pct(p)                      # '57.3%' → para el texto
    if pct_num >= 75:
        f1 = f"{fav}, favorito claro: gana {pct} de las veces."
    elif pct_num >= 65:
        f1 = f"{fav} llega como favorito sólido ({pct})."
    elif pct_num >= 57:
        f1 = f"Ligera ventaja para {fav} ({pct}), pero hay partido."
    elif pct_num >= 53:
        f1 = f"Muy parejo: {fav} apenas por delante ({pct})."
    else:
        f1 = f"Moneda al aire: {fav} apenas {pct}."

    # 2 · Elo y localía
    dif = elo_f - elo_r
    if r.fav_local and dif < 0:
        f2 = f"{riv} tiene mejor Elo ({elo_r:.0f} vs {elo_f:.0f}): la localía decide."
    elif not r.fav_local:
        f2 = f"Aun de visita, su Elo (#{rk_f}) pesa más que la localía de {riv} (#{rk_r})."
    elif dif >= 100:
        f2 = f"Brecha de {dif:.0f} puntos de Elo: #{rk_f} contra #{rk_r}."
    else:
        f2 = f"Elo #{rk_f} contra #{rk_r}, y juega en casa."

    # 3 · Récord
    rec_f, rec_r = rec.get(fav, (0, 0, 0)), rec.get(riv, (0, 0, 0))
    rf, rr = fmt_record(rec_f), fmt_record(rec_r)
    jugados_f, jugados_r = sum(rec_f), sum(rec_r)
    if jugados_f == 0 and jugados_r == 0:
        f3 = "Primer partido de la temporada para ambos."
    elif jugados_f and jugados_r and rec_f[1] == 0 and rec_r[1] == 0:
        f3 = f"Duelo de invictos: {rf} contra {rr}."
    elif jugados_f and jugados_r and rec_f[0] == 0 and rec_r[0] == 0:
        f3 = f"Ninguno ha ganado todavía ({rf} y {rr})."
    elif jugados_f and jugados_r and rec_f[1] == 0 and rec_r[0] == 0:
        f3 = f"{fav} sigue invicto ({rf}); {riv} aún no gana ({rr})."
    elif jugados_f and rec_f[1] == 0:
        f3 = f"{fav} llega invicto ({rf}); {riv} va {rr}."
    elif _pct_victorias(rec_r) > _pct_victorias(rec_f):
        f3 = f"Ojo: el récord favorece a {riv} ({rr} contra {rf})."
    elif rec_f == rec_r:
        f3 = f"Los dos llegan {rf}."
    else:
        f3 = f"Además, {fav} tiene mejor récord ({rf} contra {rr})."
    return " ".join([f1, f2, f3])


def elegir_destacados(week_games):
    """
    Los tres partidos del reel (siempre tres partidos distintos):

    El más seguro      mayor probabilidad del favorito.
    La moneda al aire  probabilidad más cercana a 50 %.
    La sorpresa        donde el modelo más contradice a Las Vegas: elige a un
                       equipo que el moneyline tiene como no favorito (el de
                       mayor brecha). Si coincide con Las Vegas en todos, el
                       partido con la mayor diferencia de probabilidad. Sin
                       líneas, el visitante con más probabilidad de ganar.
    """
    g = week_games.copy()
    seguro = g["p_fav"].idxmax()
    moneda = (g["p_home_pre"] - 0.5).abs().drop(seguro).idxmin()
    resto = g.drop([seguro, moneda])

    r = g.loc[seguro]
    out = [("El más seguro", r,
            f"El pick con más confianza de la semana:\n{r.favorito} gana con {fmt_pct(r.p_fav)}")]

    r = g.loc[moneda]
    # diferencia con los mismos % redondeados que se ven en la gráfica
    # La moneda al aire: diferencia con los mismos valores que se ven en la imagen
    a, b = fmt_par(r.p_home_pre, r.p_away_pre)
    dif = round(abs(float(a[:-1]) - float(b[:-1])), DECIMALES)
    arranque = ("Prácticamente 50-50" if dif == 0 else
                f"Solo {dif:.{DECIMALES}f} {'punto' if dif == 1 else 'puntos'} de diferencia")
    out.append(("La moneda al aire", r, f"{arranque}:\nel partido más parejo de la semana"))

    if resto.empty:
        return out
    con_linea = resto[resto["p_home_vegas"].notna()]
    if not con_linea.empty:
        pv_pick = np.where(con_linea["fav_local"], con_linea["p_home_vegas"],
                           1 - con_linea["p_home_vegas"])
        pv_pick = pd.Series(pv_pick, index=con_linea.index)
        brecha = con_linea["p_fav"] - pv_pick
        contra = brecha[pv_pick < 0.5]
        idx = contra.idxmax() if not contra.empty else brecha.abs().idxmax()
        r, pv = resto.loc[idx], pv_pick[idx]
        vf = r.favorito if pv > 0.5 else r.rival
        if pv < 0.5:
            texto = f"Las Vegas va con {vf} ({fmt_pct(1 - pv)}).\nEl modelo va con {r.favorito} ({fmt_pct(r.p_fav)})."
        elif r.p_fav > pv:
            texto = f"Las Vegas le da a {r.favorito} {fmt_pct(pv)}.\nEl modelo le tiene más fe: {fmt_pct(r.p_fav)}."
        else:
            texto = f"Las Vegas ve muy favorito a {r.favorito} ({fmt_pct(pv)}).\nEl modelo lo ve más parejo: {fmt_pct(r.p_fav)}."
    else:
        visita = resto[~resto["fav_local"]]
        idx = visita["p_fav"].idxmax() if not visita.empty else resto["p_away_pre"].idxmax()
        r = resto.loc[idx]
        texto = f"El modelo ve ganar al visitante:\n{r.away_team} con {fmt_pct(r.p_away_pre)}"
    out.append(("La sorpresa", r, texto))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Escudos (nflreadpy -> ESPN), con caché local
# ══════════════════════════════════════════════════════════════════════════════
_logo_urls = None
_logo_cache = {}
_fallos = {}


def _load_logo_urls():
    """Carga URLs de logos desde nflreadpy (solo una vez)."""
    global _logo_urls
    if _logo_urls is None:
        teams_desc = nfl.load_teams().to_pandas()
        _logo_urls = {
            "espn": dict(zip(teams_desc["team_abbr"], teams_desc["team_logo_espn"])),
            # respaldo: escudo cuadrado de nflverse (fondo sólido, se recorta en círculo)
            "squared": dict(zip(teams_desc["team_abbr"], teams_desc["team_logo_squared"])),
            # colores oficiales, para las líneas de la gráfica de evolución del Elo
            "color": dict(zip(teams_desc["team_abbr"], teams_desc["team_color"])),
            "color2": dict(zip(teams_desc["team_abbr"], teams_desc["team_color2"])),
            "division": dict(zip(teams_desc["team_abbr"], teams_desc["team_division"])),
        }
    return _logo_urls


def _luminancia(color):
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _colores_equipos(teams, respaldo=(ACENTO, ROJO, FRIO_TEXTO, TINTA, GRIS)):
    """Un color distinto por equipo: el oficial; si ya se usó o es muy claro
    para el fondo, el secundario."""
    try:
        info = _load_logo_urls()
    except Exception:  # noqa: BLE001
        info = {"color": {}, "color2": {}}
    usados, out = set(), {}
    for i, team in enumerate(teams):
        opciones = [info["color"].get(team), info["color2"].get(team), respaldo[i % len(respaldo)]]
        for c in opciones:
            if (isinstance(c, str) and len(c) == 7 and c.startswith("#")
                    and c.upper() not in usados and _luminancia(c) <= 0.7):
                break
        out[team] = c
        usados.add(c.upper())
    return out


def _descargar(url, destino, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    tmp = destino.with_suffix(".tmp")
    tmp.write_bytes(data)
    try:
        Image.open(tmp).verify()      # que sí sea una imagen
        tmp.replace(destino)
    finally:
        if tmp.exists():
            tmp.unlink()


def _archivo_logo(team, fuente):
    """Ruta local del escudo; lo descarga la primera vez."""
    carpeta = LOGOS_DIR if fuente == "espn" else LOGOS_DIR / f"_{fuente}"
    ruta = carpeta / f"{team}.png"
    if ruta.exists():
        return ruta
    if _fallos.get(fuente, 0) >= 3:            # la fuente no responde: no insistir
        return None
    url = _load_logo_urls()[fuente].get(team)
    if not isinstance(url, str) or not url:
        return None
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
        _descargar(url, ruta)
        return ruta
    except Exception as e:  # noqa: BLE001
        _fallos[fuente] = _fallos.get(fuente, 0) + 1
        if _fallos[fuente] == 1:
            print(f"  ⚠ No pude descargar escudos de {fuente} ({type(e).__name__}); uso respaldo.")
        return None


def _recortar_transparente(img):
    """Quita el borde transparente: así el tamaño depende del escudo, no del PNG."""
    img = img.convert("RGBA")
    alpha = np.asarray(img)[..., 3]
    ys, xs = np.where(alpha > 10)
    if len(xs):
        img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    return img


def _tamano_optico(w, h, px, max_w=1.45, max_h=1.12):
    """
    Tamaño en pixeles para que todos los escudos "pesen" parecido: se iguala el
    área (media geométrica = px) en vez del lado más largo. Así los escudos
    anchos (KC, SF, NE, CLE) no se ven diminutos junto a los altos (NO, LV).
    """
    a = w / h
    tw, th = px * a ** 0.5, px / a ** 0.5
    if tw > max_w * px:
        tw, th = max_w * px, max_w * px / a
    if th > max_h * px:
        tw, th = max_h * px * a, max_h * px
    return max(1, int(round(tw))), max(1, int(round(th)))


def _circulo(img):
    """Recorta un escudo cuadrado (fondo sólido) en círculo, con borde suave."""
    img = img.convert("RGBA")
    lado = min(img.size)
    img = img.crop((0, 0, lado, lado))
    esc = 4
    mask = Image.new("L", (lado * esc, lado * esc), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, lado * esc - 1, lado * esc - 1), fill=255)
    img.putalpha(mask.resize((lado, lado), Image.LANCZOS))
    return img


def obtener_logo(team, px):
    """Escudo como arreglo RGBA con tamaño óptico px (None si no hay escudo)."""
    clave = (team, px)
    if clave in _logo_cache:
        return _logo_cache[clave]
    img = None
    for fuente, preparar in (("espn", _recortar_transparente), ("squared", _circulo)):
        ruta = _archivo_logo(team, fuente)
        if ruta is None:
            continue
        try:
            img = preparar(Image.open(ruta))
            break
        except Exception:  # noqa: BLE001
            img = None
    if img is not None:
        img = np.asarray(img.resize(_tamano_optico(*img.size, px), Image.LANCZOS))
    _logo_cache[clave] = img
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 5 · Lienzo, formatos y utilidades de dibujo
# ══════════════════════════════════════════════════════════════════════════════
class Formato(NamedTuple):
    """
    Tamaño de la imagen. Todas las gráficas se diseñan en fracciones del lienzo
    vertical 1080x1920; la franja y0 → y1 de ese diseño se lleva a dest0 → dest1
    del lienzo real, y fuentes / escudos se escalan en la misma proporción.
    """
    ancho: float          # pulgadas (x DPI = pixeles)
    alto: float
    y0: float = 0.0
    y1: float = 1.0
    dest0: float = 0.0
    dest1: float = 1.0

    @property
    def k(self):
        """Factor de la coordenada y (fracción del diseño -> fracción del lienzo)."""
        return (self.dest1 - self.dest0) / (self.y1 - self.y0)

    @property
    def escala(self):
        """Factor en pixeles: fuentes, escudos, grosores."""
        return self.k * self.alto / H


FORMATOS = {
    # 1080x902 (publicación): sin zonas seguras, del pie al título del diseño vertical
    "size_1": Formato(W, 902 / DPI, y0=SAFE_BOTTOM - 0.07, y1=SAFE_TOP + 0.045,
                      dest0=0.03, dest1=0.97),
    # 1080x1920 (reel / historia): el diseño original, sin cambios
    "size_2": Formato(W, H),
}
_FMT = FORMATOS["size_2"]


@contextmanager
def usar_formato(fmt):
    """Dibuja con `fmt` dentro del bloque with."""
    global _FMT
    previo, _FMT = _FMT, fmt
    try:
        yield fmt
    finally:
        _FMT = previo


def _y(v):
    """y del diseño 1080x1920 -> y del lienzo actual (fracción de la figura)."""
    return _FMT.dest0 + (v - _FMT.y0) * _FMT.k


def _h(h):
    """Alto del diseño -> alto en el lienzo actual (fracción de la figura)."""
    return h * _FMT.k


def _fs(n):
    """Tamaño de fuente / marcador / grosor de línea escalado al formato."""
    return n * _FMT.escala


def _px(n):
    """Tamaño de escudo en pixeles escalado al formato."""
    return max(1, int(round(n * _FMT.escala)))


def _lienzo(etiqueta, titulo, pie="NFL · ELO"):
    fig = plt.figure(figsize=(_FMT.ancho, _FMT.alto), dpi=DPI, facecolor=FONDO)
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg.set_facecolor(FONDO)
    bg.set_xlim(0, _FMT.ancho)
    bg.set_ylim(0, _FMT.alto)
    for x in range(int(_FMT.ancho) + 1):
        bg.axvline(x, color=CUADRICULA, lw=0.8)
    for y in range(int(_FMT.alto) + 1):
        bg.axhline(y, color=CUADRICULA, lw=0.8)
    bg.axis("off")

    fig.text(0.5, _y(SAFE_TOP), etiqueta.upper(), ha="center", fontsize=_fs(20),
             color=ACENTO, fontweight="bold")
    fig.text(0.5, _y(SAFE_TOP - 0.035), titulo, ha="center", va="top",
             fontsize=_fs(32), fontweight="bold")
    fig.text(0.06, _y(SAFE_BOTTOM - 0.05), f"— {pie}", fontsize=_fs(14), color=GRIS)
    fig.text(0.94, _y(SAFE_BOTTOM - 0.05), "@samplesebs", fontsize=_fs(14), color=GRIS,
             ha="right")
    return fig


def _guardar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=DPI, facecolor=FONDO)
    plt.close(fig)
    return ruta


def _logo(fig, team, cx, cy, px, alpha=1.0):
    """Dibuja el escudo centrado en (cx, cy) — coordenadas de figura — con tamaño óptico px."""
    img = obtener_logo(team, px)
    alto_px, ancho_px = (px, px) if img is None else img.shape[:2]
    fig_w, fig_h = fig.get_size_inches() * fig.dpi
    w, h = ancho_px / fig_w, alto_px / fig_h
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    ax.axis("off")
    if img is None:        # sin escudo: la abreviatura
        ax.text(0.5, 0.5, team, ha="center", va="center", fontsize=px * 0.28,
                fontweight="bold", alpha=alpha, transform=ax.transAxes)
    else:
        ax.imshow(img, alpha=alpha)
    return ax


def _fig_y(ax, y):
    """Coordenada y de datos -> coordenada y de la figura."""
    pos = ax.get_position()
    y0, y1 = ax.get_ylim()
    return pos.y0 + pos.height * (y - y0) / (y1 - y0)


def _sin_bordes(ax, dejar=()):
    for lado, s in ax.spines.items():
        s.set_visible(lado in dejar)


def _zona_corta(zona):
    return ZONAS_CORTAS.get(zona, zona.split("/")[-1].replace("_", " "))


def _fmt_dia(ts):
    return f"{DIAS[ts.weekday()]} {ts.day} {MESES[ts.month - 1]}"


def _fmt_fecha_hora(ts, zona):
    if pd.isna(ts):
        return ""
    return f"{_fmt_dia(ts)} · {ts:%H:%M} h ({_zona_corta(zona)})"


# ══════════════════════════════════════════════════════════════════════════════
# 6 · Gráficas
# ══════════════════════════════════════════════════════════════════════════════
def graf_aciertos_semana(tabla, threshold, ruta):
    """Barras = % de aciertos de cada semana · línea = % acumulado.
    Los conteos van como récord de la NFL: aciertos-fallos(-empates)."""
    t = tabla.reset_index(drop=True)
    u = t.iloc[-1]
    fig = _lienzo("La prueba", "¿Cómo va el modelo?")

    fig.text(0.5, _y(0.715), record_af(u.acum_aciertos, u.acum_picks, u.acum_empates),
             ha="center", va="center", fontsize=_fs(96), fontweight="bold", color=ACENTO)
    etiqueta = ("RÉCORD DEL MODELO (ACIERTOS–FALLOS–EMPATES)" if u.acum_empates
                else "RÉCORD DEL MODELO (ACIERTOS–FALLOS)")
    fig.text(0.5, _y(0.655), etiqueta, ha="center", fontsize=_fs(18))
    fig.text(0.5, _y(0.632), f"{u.acum_pct_aciertos:.0%} de aciertos en {int(u.acum_picks)} partidos",
             ha="center", fontsize=_fs(15), color=GRIS)
    if threshold > 0.5:
        fig.text(0.5, _y(0.612), f"Solo cuentan los picks con probabilidad > {threshold:.0%}",
                 ha="center", fontsize=_fs(13), color=GRIS)

    ax = fig.add_axes([0.10, _y(0.34), 0.80, _h(0.25)], facecolor="none")
    n = len(t)
    x = np.arange(n)
    pct = t["pct_aciertos"].fillna(0).to_numpy()
    ax.bar(x, pct, width=0.62, color=FRIO, zorder=2)

    # récord de la semana (aciertos-fallos) dentro de cada barra, en la base
    vertical = n > 9
    fs = 17 if n <= 6 else (15 if n <= 9 else 12)
    for xi, r in zip(x, t.itertuples()):
        ax.text(xi, 0.03, record_af(r.aciertos, r.picks, r.empates), ha="center", va="bottom",
                fontsize=_fs(fs), fontweight="bold", rotation=90 if vertical else 0, zorder=5)

    acum = t["acum_pct_aciertos"].to_numpy()
    ax.plot(x, acum, color=ACENTO, lw=_fs(4), marker="o", ms=_fs(11), zorder=6,
            markeredgecolor=FONDO, markeredgewidth=_fs(2))
    ax.annotate(f"{acum[-1]:.0%}", (x[-1], acum[-1]), xytext=(0, _fs(16)),
                textcoords="offset points", ha="center", fontsize=_fs(20),
                fontweight="bold", color=ACENTO, zorder=7)

    pad = max(0.0, (6 - n) / 2)
    ax.set_xlim(-0.6 - pad, n - 0.4 + pad)
    ax.set_ylim(0, min(1.15, max(0.8, float(np.nanmax(np.r_[pct, acum])) + 0.2)))
    ax.axhline(0.5, color=TINTA, lw=_fs(1.2), ls="--", alpha=0.6, zorder=3)
    fig.text(0.095, _fig_y(ax, 0.5), "50%", ha="right", va="center", fontsize=_fs(13), color=GRIS)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(w)}" for w in t["semana"]], fontsize=_fs(16 if n <= 10 else 12))
    ax.tick_params(axis="x", length=0, pad=_fs(8))
    ax.set_yticks([])
    _sin_bordes(ax, dejar=("bottom",))
    ax.spines["bottom"].set_color(GRIS)

    fig.text(0.49, _y(0.285), "■ récord de la semana", ha="right", fontsize=_fs(15),
             color=FRIO_TEXTO)
    fig.text(0.53, _y(0.285), "● % acumulado", ha="left", fontsize=_fs(15), color=ACENTO)
    fig.text(0.5, _y(0.262), "- - -  50 % = moneda al aire", ha="center", fontsize=_fs(13),
             color=GRIS)

    comp = ["Siempre el local: "
            f"{record_af(u.acum_siempre_local_aciertos, u.acum_picks, u.acum_empates)}"]
    if MOSTRAR_VEGAS and u.acum_vegas_partidos:
        comp.append("Favorito de Las Vegas: "
                    f"{record_af(u.acum_vegas_aciertos, u.acum_vegas_partidos, u.acum_vegas_empates)}")
    fig.text(0.5, _y(0.228), "   ·   ".join(comp), ha="center", fontsize=_fs(14), color=GRIS)
    return _guardar(fig, ruta)


def graf_calibracion(calib, ruta, alta=0.70):
    """Barras = % real que ganó el favorito en cada rango de confianza ·
    raya = lo que el modelo prometió (prob. media). Arriba, confiar vs no confiar."""
    c = calib[calib["partidos"] > 0].reset_index(drop=True)
    fig = _lienzo("La prueba", "¿Cuándo hay que creerle?")
    fig.text(0.5, _y(0.757), "Si el modelo dice 70 %, ¿el favorito gana 70 % de las veces?",
             ha="center", fontsize=_fs(15), color=GRIS)

    def _resumen(filas):
        n, a, e = filas["partidos"].sum(), filas["aciertos"].sum(), filas["empates"].sum()
        return (a + 0.5 * e) / n if n else np.nan, record_af(a, n, e)

    lim = [float(r.split("–")[0].rstrip("%+")) / 100 for r in c["rango"].astype(str)]
    conf = c[[v >= alta for v in lim]]
    dudas = c.iloc[:1]
    for cx, filas, texto, color in ((0.29, conf, f"≥{alta:.0%} seguro", ACENTO),
                                    (0.71, dudas, f"{dudas['rango'].iloc[0]} seguro"
                                     if len(dudas) else "", ROJO)):
        pct, rec = _resumen(filas)
        if np.isnan(pct):
            continue
        color = color if pct >= 0.5 else ROJO
        fig.text(cx, _y(0.698), f"{pct:.0%}", ha="center", va="center", fontsize=_fs(80),
                 fontweight="bold", color=color)
        fig.text(cx, _y(0.662), f"ACIERTA CUANDO ESTÁ\n{texto.upper()}", ha="center", va="top",
                 fontsize=_fs(15), linespacing=1.4)
        fig.text(cx, _y(0.615), rec, ha="center", fontsize=_fs(15), color=GRIS)

    ax = fig.add_axes([0.10, _y(0.33), 0.80, _h(0.26)], facecolor="none")
    n = len(c)
    x = np.arange(n)
    real = c["pct_real"].to_numpy()
    prom = c["prob_media"].to_numpy()
    colores = [ROJO if r < 0.5 else (ACENTO if r >= p - 0.02 else FRIO)
               for r, p in zip(real, prom)]
    ax.bar(x, real, width=0.62, color=colores, zorder=2)
    for xi, r, p, rec in zip(x, real, prom, c["record"]):
        ax.text(xi, 0.03, rec, ha="center", va="bottom", fontsize=_fs(16),
                fontweight="bold", color=TINTA if r < 0.5 or r < p - 0.02 else FONDO, zorder=5)
        ax.text(xi, max(r, p) + 0.03, f"{r:.0%}", ha="center", va="bottom", fontsize=_fs(20),
                fontweight="bold", color=ROJO if r < 0.5 else TINTA, zorder=7)
        ax.plot([xi - 0.38, xi + 0.38], [p, p], color=TINTA, lw=_fs(3.5),
                solid_capstyle="round", zorder=6)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, color=TINTA, lw=_fs(1.2), ls="--", alpha=0.6, zorder=3)
    fig.text(0.095, _fig_y(ax, 0.5), "50%", ha="right", va="center", fontsize=_fs(13), color=GRIS)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{int(p)} partidos" for r, p in zip(c["rango"], c["partidos"])],
                       fontsize=_fs(15), linespacing=1.5)
    ax.tick_params(axis="x", length=0, pad=_fs(8))
    ax.set_yticks([])
    _sin_bordes(ax, dejar=("bottom",))
    ax.spines["bottom"].set_color(GRIS)

    fig.text(0.49, _y(0.262), "■ lo que acertó", ha="right", fontsize=_fs(15), color=ACENTO)
    fig.text(0.53, _y(0.262), "▬ lo que prometió", ha="left", fontsize=_fs(15))
    fig.text(0.5, _y(0.240), "- - -  50 % = moneda al aire", ha="center", fontsize=_fs(13),
             color=GRIS)
    return _guardar(fig, ruta)


def graf_suben_caen(cambios, contexto, etiqueta, titulo, ruta, n=TOP_N):
    """Top n que más ELO ganan y top n que más pierden, con escudo."""
    df = cambios.dropna().sort_values()
    sel = pd.concat([df.head(n), df.tail(n)])
    fig = _lienzo(etiqueta, titulo)
    ax = fig.add_axes([0.25, _y(0.26), 0.67, _h(0.44)], facecolor="none")
    y = np.arange(len(sel))
    colores = [ACENTO if c > 0 else (ROJO if c < 0 else GRIS) for c in sel]
    ax.barh(y, sel.to_numpy(), color=colores, height=0.65)
    ax.axvline(0, color=TINTA, lw=_fs(1.5))
    lim = max(float(sel.abs().max()), 1.0) * 1.45
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, len(sel) - 0.4)
    for yi, (team, c) in zip(y, sel.items()):
        der = c >= 0
        ax.text(c + (lim * 0.03 if der else -lim * 0.03), yi, f"{c:+.0f}", va="center",
                ha="left" if der else "right", fontsize=_fs(18), fontweight="bold")
        if contexto:   # resultado/récord del lado contrario a la barra
            ax.text(-lim * 0.03 if der else lim * 0.03, yi, contexto.get(team, "Descansó"),
                    va="center", ha="right" if der else "left", fontsize=_fs(12), color=GRIS)
        fy = _fig_y(ax, yi)
        _logo(fig, team, 0.085, fy, _px(58))
        fig.text(0.14, fy, team, va="center", fontsize=_fs(20))
    ax.set_xticks([])
    ax.set_yticks([])
    _sin_bordes(ax)
    return _guardar(fig, ruta)


def graf_top(elo_actual, ruta, n=TOP_N):
    """Ranking del modelo — el diseño de referencia, ahora con escudo."""
    df = elo_actual.sort_values(ascending=False).head(n)[::-1]
    fig = _lienzo("Ranking del modelo", f"Top {n} según el Elo")
    ax = fig.add_axes([0.24, _y(0.30), 0.60, _h(0.38)], facecolor="none")
    y = np.arange(n)
    base = 1400
    ax.barh(y, df.to_numpy() - base, left=base, color=[FRIO] * (n - 1) + [ACENTO], height=0.6)
    for yi, v in zip(y, df.to_numpy()):
        ax.text(v + 5, yi, f"{v:.0f}", va="center", fontsize=_fs(22), fontweight="bold")
    ax.set_xlim(base, df.max() + 60)
    ax.set_ylim(-0.53, n - 0.47)
    for yi, team in zip(y, df.index):
        fy = _fig_y(ax, yi)
        _logo(fig, team, 0.085, fy, _px(64))
        fig.text(0.145, fy, team, va="center", fontsize=_fs(24))
    ax.set_xticks([])
    ax.set_yticks([])
    _sin_bordes(ax)
    fig.text(0.5, _y(0.26), "Promedio de la liga ≈ 1500", ha="center", fontsize=_fs(16),
             color=GRIS)
    return _guardar(fig, ruta)


def _separar(ys, gap):
    """Reparte posiciones (1-D) para que queden al menos `gap` entre sí, sin perder el orden."""
    orden = np.argsort(ys)[::-1]                 # de arriba hacia abajo
    out = np.asarray(ys, dtype=float).copy()
    for a, b in zip(orden[:-1], orden[1:]):
        out[b] = min(out[b], out[a] - gap)
    exceso = out.min() - min(ys)                 # si se fue muy abajo, centrar el bloque
    if exceso < 0:
        out -= exceso / 2
    return out


def graf_evolucion_elo(elo_tab, semana, ruta, n=TOP_N):
    """Elo de los 32 equipos semana a semana desde la pretemporada.
    Todos en gris; el top n actual y el que más cae, en color con su escudo."""
    cols = [c for c in elo_tab.columns if c <= semana]
    t = elo_tab[cols].dropna()
    actual, inicio = t[cols[-1]], t[cols[0]]
    destacados = list(actual.sort_values(ascending=False).index[:n])
    cae = (actual - inicio).idxmin()
    if cae not in destacados:
        destacados.append(cae)
    colores = _colores_equipos(destacados)

    fig = _lienzo("Evolución del Elo", "Así se mueve la liga")
    fig.text(0.5, _y(0.762), f"De la pretemporada a la semana {semana - 1}",
             ha="center", fontsize=_fs(15), color=GRIS)
    ax = fig.add_axes([0.13, _y(0.27), 0.64, _h(0.45)], facecolor="none")
    x = np.arange(len(cols))
    for team, fila in t.iterrows():
        if team not in destacados:
            ax.plot(x, fila.to_numpy(), color=GRIS, lw=_fs(1.5), alpha=0.3, zorder=2)
    for z, team in enumerate(destacados[::-1]):    # el #1 queda encima
        y = t.loc[team].to_numpy()
        ax.plot(x, y, color=colores[team], lw=_fs(4), zorder=4 + z, solid_capstyle="round")
        ax.plot(x[-1], y[-1], "o", ms=_fs(11), color=colores[team], zorder=4 + z,
                markeredgecolor=FONDO, markeredgewidth=_fs(2))

    lo, hi = float(t.min().min()), float(t.max().max())
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.25, len(cols) - 0.75)
    ax.axhline(1500, color=TINTA, lw=_fs(1.2), ls="--", alpha=0.6, zorder=3)
    ax.text(0.02, 1500, "promedio ≈ 1500", transform=ax.get_yaxis_transform(),
            va="center", fontsize=_fs(12), color=GRIS, zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=FONDO, edgecolor="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(["Inicio"] + [f"S{c - 1}" for c in cols[1:]],
                       fontsize=_fs(15 if len(cols) <= 10 else 11))
    ax.tick_params(axis="x", length=0, pad=_fs(8))
    paso = 50 if hi - lo < 300 else 100
    ax.set_yticks(np.arange(np.ceil((lo - pad) / paso) * paso, hi + pad, paso))
    ax.tick_params(axis="y", length=0, labelsize=_fs(13), labelcolor=GRIS)
    _sin_bordes(ax, dejar=("bottom",))
    ax.spines["bottom"].set_color(GRIS)

    # Escudo + Elo al final de cada línea destacada, sin encimarse
    fy = [_fig_y(ax, actual[team]) for team in destacados]
    for team, y in zip(destacados, _separar(fy, _h(0.034))):
        _logo(fig, team, 0.815, y, _px(46))
        fig.text(0.85, y, f"{actual[team]:.0f}", va="center", fontsize=_fs(16),
                 fontweight="bold", color=colores[team])

    fig.text(0.5, _y(0.215), f"En color: top {n} y el que más cae · en gris: el resto",
             ha="center", fontsize=_fs(13), color=GRIS)
    return _guardar(fig, ruta)


DIVISIONES = {"East": "Este", "North": "Norte", "South": "Sur", "West": "Oeste"}


def graf_evolucion_divisiones(elo_tab, semana, ruta):
    """Elo semana a semana en 8 paneles (AFC a la izquierda, NFC a la derecha),
    los 4 equipos de cada división en color y con la misma escala en todos."""
    cols = [c for c in elo_tab.columns if c <= semana]
    t = elo_tab[cols].dropna()
    division = _load_logo_urls()["division"]
    lo, hi = float(t.min().min()), float(t.max().max())
    pad = (hi - lo) * 0.08
    x = np.arange(len(cols))

    fig = _lienzo("Evolución del Elo", "Así va cada división")
    fig.text(0.5, _y(0.762), f"De la pretemporada a la semana {semana - 1}",
             ha="center", fontsize=_fs(15), color=GRIS)
    top, alto_fila = _y(0.735), _h(0.12)
    for col, conf in enumerate(("AFC", "NFC")):
        x0 = 0.07 + 0.47 * col
        for fila, (div_en, div_es) in enumerate(DIVISIONES.items()):
            y_fila = top - fila * alto_fila
            fig.text(x0, y_fila, f"{conf} {div_es}".upper(), va="top", fontsize=_fs(13),
                     fontweight="bold", color=ACENTO)
            equipos = [e for e in t.index if division.get(e) == f"{conf} {div_en}"]
            equipos.sort(key=lambda e: -t.loc[e, cols[-1]])
            colores = _colores_equipos(equipos)

            ax = fig.add_axes([x0, y_fila - alto_fila + _h(0.012), 0.30, alto_fila - _h(0.035)],
                              facecolor="none")
            ax.axhline(1500, color=TINTA, lw=_fs(1), ls="--", alpha=0.5, zorder=1)
            for z, e in enumerate(equipos[::-1]):
                y = t.loc[e].to_numpy()
                ax.plot(x, y, color=colores[e], lw=_fs(3), zorder=2 + z, solid_capstyle="round")
                ax.plot(x[-1], y[-1], "o", ms=_fs(7), color=colores[e], zorder=2 + z,
                        markeredgecolor=FONDO, markeredgewidth=_fs(1.5))
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.2, len(cols) - 0.8)
            ax.set_yticks([])
            ax.set_xticks(x)
            ultima = fila == len(DIVISIONES) - 1
            ax.set_xticklabels(["Inicio"] + [f"S{c - 1}" for c in cols[1:]] if ultima else [],
                               fontsize=_fs(11 if len(cols) <= 8 else 8))
            ax.tick_params(axis="x", length=0, pad=_fs(4))
            _sin_bordes(ax, dejar=("bottom",))
            ax.spines["bottom"].set_color(CUADRICULA)

            fy = [_fig_y(ax, t.loc[e, cols[-1]]) for e in equipos]
            for e, y in zip(equipos, _separar(fy, _h(0.0175))):
                _logo(fig, e, x0 + 0.325, y, _px(30))
                fig.text(x0 + 0.35, y, f"{t.loc[e, cols[-1]]:.0f}", va="center",
                         fontsize=_fs(11), fontweight="bold", color=colores[e])

    fig.text(0.5, _y(0.215), "Misma escala en todas las divisiones · - - - 1500 = promedio",
             ha="center", fontsize=_fs(13), color=GRIS)
    return _guardar(fig, ruta)


def graf_picks(week_games, semana, threshold, ruta, zona=ZONA_HORARIA):
    """Picks en orden de juego (jueves → lunes), escudos en lugar de nombres."""
    g = week_games.sort_values(["kickoff", "game_id"], na_position="last")
    fig = _lienzo(f"Semana {semana}", "Mis pronósticos")
    fig.text(0.5, _y(0.762), f"En orden de juego · hora {_zona_corta(zona)}",
             ha="center", fontsize=_fs(15), color=GRIS)

    # Renglones: encabezado de día + un renglón por partido
    filas, dia_actual = [], "∅"
    for r in g.itertuples():
        dia = r.kickoff.date() if pd.notna(r.kickoff) else None
        if dia != dia_actual:
            filas.append(("dia", r))
            dia_actual = dia
        filas.append(("juego", r))
    n_dias = sum(t == "dia" for t, _ in filas)
    # Encabezados de columna
    fig.text(0.215, _y(0.735), "EQUIPO GANADOR", ha="center", va="center", fontsize=_fs(12),
             fontweight="bold")
    fig.text(0.72, _y(0.735), "PROBABILIDAD DE VICTORIA", ha="center", va="center",
             fontsize=_fs(12), fontweight="bold")
    fig.add_artist(plt.Line2D([0.075, 0.925], [_y(0.724), _y(0.724)], color=GRIS, lw=_fs(1)))

    H_DIA, GAP = 0.95, 0.35
    unidades = (len(filas) - n_dias) + H_DIA * n_dias + GAP * (n_dias - 1)
    top, bottom = _y(0.715), _y(0.228)
    u = (top - bottom) / unidades
    u_px = u * fig.get_size_inches()[1] * fig.dpi
    fav_px = int(min(_px(60), u_px * 0.84))
    riv_px = int(fav_px * 0.75)

    y, primero, sin_pick, hora_prev = top, True, False, None
    for tipo, r in filas:
        if tipo == "dia":
            if not primero:
                y -= GAP * u
            primero, hora_prev = False, None
            texto = _fmt_dia(r.kickoff).upper() if pd.notna(r.kickoff) else "POR DEFINIR"
            fig.text(0.075, y - H_DIA * u / 2, texto, fontsize=_fs(14), fontweight="bold",
                     color=ACENTO, va="center")
            y -= H_DIA * u
            continue

        yc = y - u / 2
        es_pick = r.p_fav > threshold
        sin_pick |= not es_pick
        hora = f"{r.kickoff:%H:%M}" if pd.notna(r.kickoff) else ""
        if hora != hora_prev:          # la hora solo cuando cambia
            fig.text(0.075, yc, hora, fontsize=_fs(13), color=GRIS, va="center")
            hora_prev = hora
        _logo(fig, r.favorito, 0.215, yc, fav_px)
        fig.text(0.29, yc, "vs", fontsize=_fs(13), color=GRIS, va="center", ha="center")
        _logo(fig, r.rival, 0.35, yc, riv_px, alpha=0.6)
        fig.text(0.405, yc, "en casa" if r.fav_local else "de visita",
                 fontsize=_fs(12), color=GRIS, va="center")

        bar = fig.add_axes([0.52, yc - u * 0.24, 0.285, u * 0.48], facecolor="none")
        color = ACENTO if r.p_fav >= ALTA_CONFIANZA else FRIO
        bar.barh([0], [r.p_fav], color=color, height=1)
        bar.barh([0], [1 - r.p_fav], left=r.p_fav, color=PISTA, height=1)
        bar.set_xlim(0, 1)
        bar.set_ylim(-0.5, 0.5)
        bar.axis("off")
        fig.text(0.83, yc, fmt_pct(r.p_fav), fontsize=_fs(17), fontweight="bold", va="center",
                 color=TINTA if es_pick else GRIS)
        y -= u

    leyenda = f"■ confianza alta (≥ {ALTA_CONFIANZA:.0%})     ■ confianza media"
    fig.text(0.5, _y(0.205), leyenda, ha="center", fontsize=_fs(13), color=GRIS)
    if sin_pick:
        fig.text(0.5, _y(0.188), f"En gris: sin pick (≤ {threshold:.0%})", ha="center",
                 fontsize=_fs(12), color=GRIS)
    return _guardar(fig, ruta)


def _envolver(texto, ancho=44):
    """Parte el texto en renglones de ~ancho caracteres (respeta los \\n que ya traiga)."""
    return "\n".join(textwrap.fill(p, ancho) for p in texto.split("\n"))


def graf_duelo(r, etiqueta, explicacion, ruta, zona=ZONA_HORARIA, rec=None, rank=None):
    """Un partido: récord, escudos, barra de probabilidad y el comentario."""
    rec, rank = rec or {}, rank or {}
    fig = _lienzo(etiqueta, f"{r.away_team} @ {r.home_team}")
    fig.text(0.5, _y(0.762), _fmt_fecha_hora(r.kickoff, zona), ha="center",
             fontsize=_fs(16), color=GRIS)

    p_loc, p_vis = float(r.p_home_pre), float(r.p_away_pre)
    fav_vis = p_vis > p_loc
    for team, cx in ((r.away_team, 0.27), (r.home_team, 0.73)):
        if team in rec:   # récord arriba del escudo
            fig.text(cx, _y(0.725), "RÉCORD", ha="center", fontsize=_fs(12), color=GRIS,
                     fontweight="bold")
            fig.text(cx, _y(0.700), fmt_record(rec[team]), ha="center", va="center",
                     fontsize=_fs(28), fontweight="bold")
        _logo(fig, team, cx, _y(0.600), _px(240))

    ax = fig.add_axes([0.08, _y(0.455), 0.84, _h(0.06)], facecolor="none")
    ax.barh([0], [p_vis], color=ACENTO if fav_vis else FRIO, height=1)
    ax.barh([0], [p_loc], left=p_vis, color=FRIO if fav_vis else ACENTO, height=1)
    ax.axvline(0.5, color=TINTA, lw=_fs(1.5), ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")

    txt_vis, txt_loc = fmt_par(p_vis, p_loc)

    fig.text(0.08, _y(0.435), txt_vis, fontsize=_fs(40), fontweight="bold", va="top",
         color=ACENTO if fav_vis else GRIS)
    fig.text(0.92, _y(0.435), txt_loc, fontsize=_fs(40), fontweight="bold", va="top",
            ha="right", color=GRIS if fav_vis else ACENTO)

    def _rk(team):
        return f" (#{rank[team]})" if team in rank else ""

    fig.text(0.08, _y(0.372), f"visitante · Elo {r.elo_away_pre:.0f}{_rk(r.away_team)}",
             fontsize=_fs(16), color=GRIS)
    fig.text(0.92, _y(0.372), f"local · Elo {r.elo_home_pre:.0f}{_rk(r.home_team)}",
             fontsize=_fs(16), color=GRIS, ha="right")
    texto = _envolver(explicacion)
    fs = 19 if texto.count("\n") < 3 else 18
    fig.text(0.5, _y(0.285), texto, ha="center", va="center", fontsize=_fs(fs), linespacing=1.55)
    return _guardar(fig, ruta)


# ══════════════════════════════════════════════════════════════════════════════
# 7 · Resumen en texto (para el guion)
# ══════════════════════════════════════════════════════════════════════════════
def _pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


def escribir_resumen(ruta, semana, threshold, tabla, calib, ev, week_games, destacados, zona):
    L = [f"# Semana {semana} · Pronosticando la NFL con un modelo de ELO", ""]
    if len(tabla):
        u = tabla.iloc[-1]
        w = int(u.semana)
        L += [f"## ¿Cómo va el modelo? (semanas 1–{w} · pick si prob. > {threshold:.0%})", "",
              f"- Récord del modelo en la temporada (aciertos-fallos-empates): "
              f"**{u.acum_record} ({_pct(u.acum_pct_aciertos)})**",
              f"- Semana {w}: {u.record} ({_pct(u.pct_aciertos)})",
              f"- Aciertos que el propio modelo esperaba: {u.acum_aciertos_esperados:.1f} "
              f"→ reales: {u.acum_aciertos + 0.5 * u.acum_empates:g} (empate = ½)",
              f"- Brier: {u.acum_brier:.3f} (moneda al aire = 0.250) · "
              f"Log loss: {u.acum_log_loss:.3f} (moneda = 0.693)",
              f"- Cuando eligió al local: "
              f"{record_af(u.acum_aciertos_local, u.acum_picks_local, u.acum_empates_local)} · "
              f"al visitante: "
              f"{record_af(u.acum_aciertos_visita, u.acum_picks_visita, u.acum_empates_visita)}",
              f"- En esos mismos partidos, siempre el local: "
              f"{record_af(u.acum_siempre_local_aciertos, u.acum_picks, u.acum_empates)}"]
        if u.acum_vegas_partidos:
            L += [f"- Favorito de Las Vegas: "
                  f"{record_af(u.acum_vegas_aciertos, u.acum_vegas_partidos, u.acum_vegas_empates)} "
                  f"({_pct(u.acum_vegas_pct)}) · el modelo coincide con Las Vegas en "
                  f"{_pct(u.acum_acuerdo_vegas)} de los picks · Brier Las Vegas: {u.acum_vegas_brier:.3f}"]
        ult = ev[(ev["week"] == w) & ~ev["empate"]]       # el batacazo necesita ganador
        if len(ult):
            b = ult.sort_values("p_ganador").iloc[0]
            L += [f"- El batacazo de la semana {w}: **{b.ganador}** ganó con solo "
                  f"{b.p_ganador:.0%} de probabilidad ({b.away_team} {int(b.away_score)}–"
                  f"{int(b.home_score)} {b.home_team})"]
        L += ["", "### Calibración (temporada)", "",
              "| Confianza | Partidos | Récord | Prob. media | % real |",
              "|---|---|---|---|---|"]
        for c in calib.itertuples():
            if c.partidos:
                L.append(f"| {c.rango} | {c.partidos} | {c.record} | "
                         f"{c.prob_media:.0%} | {c.pct_real:.0%} |")
        L.append("")

    L += [f"## Picks de la semana {semana} (hora {_zona_corta(zona)})", ""]
    for r in week_games.sort_values(["kickoff", "game_id"], na_position="last").itertuples():
        cuando = f"{_fmt_dia(r.kickoff)} {r.kickoff:%H:%M}" if pd.notna(r.kickoff) else "Por definir"
        marca = "" if r.p_fav > threshold else " (sin pick)"
        L.append(f"- {cuando} · {r.away_team} @ {r.home_team} → **{r.favorito} {r.p_fav:.0%}**{marca}")

    L += ["", "## Destacados", ""]
    for etiqueta, r, texto in destacados:
        L.append(f"- **{etiqueta}** · {r.away_team} @ {r.home_team}: {texto.replace(chr(10), ' ')}")
    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")
    return ruta


def escribir_comentarios(ruta, semana, partidos, rec, rank, zona):
    """comentarios.md — texto listo para copiar en cada historia."""
    L = [f"# Semana {semana} · Historias por partido", ""]
    for i, r, texto in partidos:
        L += [f"## {i:02d} · {r.away_team} @ {r.home_team} — {_fmt_fecha_hora(r.kickoff, zona)}", "",
              f"Pick: **{r.favorito} {r.p_fav:.0%}** · Elo {r.elo_away_pre:.0f} "
              f"(#{rank.get(r.away_team, '–')}) vs {r.elo_home_pre:.0f} "
              f"(#{rank.get(r.home_team, '–')}) · Récord {r.away_team} "
              f"{fmt_record(rec.get(r.away_team, (0, 0, 0)))}, {r.home_team} "
              f"{fmt_record(rec.get(r.home_team, (0, 0, 0)))}", "",
              texto, ""]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(L), encoding="utf-8")
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
# 8 · Orquestador
# ══════════════════════════════════════════════════════════════════════════════
def _slug(texto):
    tabla = str.maketrans("áéíóúñ ", "aeioun_")
    return texto.lower().translate(tabla)


def generar_contenido(season=SEASON, threshold=THRESHOLD, semana=None, actualizar=True,
                      out_dir=None, zona=ZONA_HORARIA, logos_dir=None, top_n=TOP_N,
                      historias=True, formatos=None):
    """
    Corre todo el flujo semanal y regresa un dict con los DataFrames y las rutas.

    semana    : semana de los picks. None = la próxima por jugarse. Las métricas
                usan solo las semanas anteriores (lo que se sabía en ese momento).
    historias : True = una imagen por partido en partidos/ + comentarios.md.
    formatos  : tamaños a generar (claves de FORMATOS). None = todos.
    """
    global LOGOS_DIR
    if logos_dir:
        LOGOS_DIR = Path(logos_dir)
    if not 0.5 <= threshold < 1:
        raise ValueError("threshold debe estar entre 0.50 y 1 (ej. 0.5, 0.6)")

    print("── 1 · Datos " + "─" * 60)
    schedule, final_elos = cargar_datos(season)
    games_df, elo_evolution_df = calcular_elo(schedule, final_elos)
    if actualizar:
        guardar_archivos(games_df, elo_evolution_df, season)
    games = enriquecer(games_df, schedule, zona)

    if semana is None:
        semana = int(current_week_probabilities(games_df)["week"].iloc[0])
    elo_tab = tabla_elo(elo_evolution_df)
    if semana not in elo_tab.columns:
        raise ValueError(f"La semana {semana} no está en el calendario de {season}")
    carpeta = Path(out_dir) if out_dir else REELS_DIR / f"semana_{semana:02d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"Semana de los picks : {semana}  →  {carpeta}")

    print("── 2 · Métricas " + "─" * 57)
    ev = evaluar_partidos(games[games["week"] < semana], threshold)
    tabla = metricas_por_semana(ev) if len(ev) else pd.DataFrame()
    calib = calibracion(ev)
    ev.to_csv(carpeta / "evaluacion_partidos.csv", index=False)
    tabla.to_csv(carpeta / "metricas_semanales.csv", index=False)
    calib.to_csv(carpeta / "calibracion.csv", index=False)
    if len(tabla):
        u = tabla.iloc[-1]
        print(f"Récord acumulado    : {u.acum_record} "
              f"({_pct(u.acum_pct_aciertos)}) · threshold > {threshold:.0%}")
        print(f"Semana {int(u.semana)}            : {u.record} "
              f"({_pct(u.pct_aciertos)})")
        print(f"Brier / Log loss    : {u.acum_brier:.3f} / {u.acum_log_loss:.3f}")
    else:
        print("Aún no hay partidos jugados antes de esta semana.")

    print("── 3 · Gráficas " + "─" * 57)
    week_games = games[games["week"] == semana]
    destacados = elegir_destacados(week_games)
    rec = records_equipos(games, semana)
    rank = elo_tab[semana].rank(ascending=False, method="min").astype(int).to_dict()
    rutas = []
    # Las mismas imágenes en cada tamaño: reels/semana_XX/size_1, size_2, ...
    for nombre_fmt in formatos or FORMATOS:
        img = carpeta / nombre_fmt
        with plt.rc_context(ESTILO), usar_formato(FORMATOS[nombre_fmt]):
            if len(tabla):
                rutas.append(graf_aciertos_semana(tabla, threshold, img / "01_aciertos_por_semana.png"))
            if semana > 1:
                rutas.append(graf_suben_caen(
                    elo_tab[semana] - elo_tab[semana - 1], resultados_semana(games, semana - 1),
                    "Quién sube, quién cae", f"Cambio de Elo\nen la semana {semana - 1}",
                    img / f"02_elo_semana_{semana - 1:02d}.png", n=top_n))
            if semana > 2:   # en la semana 2 sería idéntica a la anterior
                rutas.append(graf_suben_caen(
                    elo_tab[semana] - elo_tab[elo_tab.columns.min()], records(games, semana),
                    "Quién sube, quién cae", "Cambio de Elo\ndesde la pretemporada",
                    img / "03_elo_temporada.png", n=top_n))
            rutas.append(graf_top(elo_tab[semana], img / "04_top_elo.png", n=top_n))
            rutas.append(graf_picks(week_games, semana, threshold,
                                    img / f"{semana - 1:02d}_picks_semana_{semana:02d}.png", zona))
            for i, (etiqueta, r, texto) in enumerate(destacados, start=6):
                nombre = f"{i:02d}_{_slug(etiqueta)}_{r.away_team}_{r.home_team}.png"
                rutas.append(graf_duelo(r, etiqueta, texto, img / nombre, zona, rec, rank))
            if semana > 1:
                rutas.append(graf_evolucion_elo(elo_tab, semana, img / "09_evolucion_elo.png",
                                                n=top_n))
                rutas.append(graf_evolucion_divisiones(elo_tab, semana,
                                                       img / "10_evolucion_divisiones.png"))
            if calib["partidos"].sum():
                rutas.append(graf_calibracion(calib, img / "11_calibracion.png"))

            # Historias: una imagen por partido, en orden de juego, con su comentario
            if historias:
                orden = week_games.sort_values(["kickoff", "game_id"], na_position="last")
                n, partidos = len(orden), []
                for i, (_, r) in enumerate(orden.iterrows(), start=1):
                    texto = comentario_partido(r, rec, rank)
                    partidos.append((i, r, texto))
                    rutas.append(graf_duelo(
                        r, f"Semana {semana} · Partido {i} de {n}", texto,
                        img / "partidos" / f"{i:02d}_{r.away_team}_{r.home_team}.png",
                        zona, rec, rank))
                rutas.append(escribir_comentarios(img / "partidos" / "comentarios.md",
                                                  semana, partidos, rec, rank, zona))
    rutas.append(escribir_resumen(carpeta / "resumen.md", semana, threshold, tabla, calib,
                                  ev, week_games, destacados, zona))
    for r in rutas:
        print("✓", r.relative_to(ROOT) if r.is_relative_to(ROOT) else r)

    return {"semana": semana, "games": games, "evaluacion": ev, "metricas": tabla,
            "calibracion": calib, "destacados": destacados, "carpeta": carpeta, "rutas": rutas}


def main(argv=None):
    p = argparse.ArgumentParser(description="Contenido semanal · NFL con ELO")
    p.add_argument("--season", type=int, default=SEASON)
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="Prob. mínima (estrictamente mayor) para contar un pick. Default 0.50")
    p.add_argument("--semana", type=int, default=None,
                   help="Semana de los picks. Default: la próxima por jugarse")
    p.add_argument("--sin-actualizar", action="store_true",
                   help="No reescribe los CSV de data/processed")
    p.add_argument("--out", default=None, help="Carpeta de salida (default: reels/semana_XX)")
    p.add_argument("--zona", default=ZONA_HORARIA, help="Zona horaria de los horarios")
    p.add_argument("--logos-dir", default=None, help="Carpeta de caché de escudos")
    p.add_argument("--top", type=int, default=TOP_N, help="Equipos por lado en las gráficas de ELO")
    p.add_argument("--sin-historias", action="store_true",
                   help="No genera la imagen de cada partido (carpeta partidos/)")
    p.add_argument("--formatos", nargs="+", choices=list(FORMATOS), default=None,
                   help="Tamaños a generar: size_1 (1080x902), size_2 (1080x1920). Default: todos")
    a = p.parse_args(argv)
    plt.switch_backend("Agg")
    generar_contenido(season=a.season, threshold=a.threshold, semana=a.semana,
                      actualizar=not a.sin_actualizar, out_dir=a.out, zona=a.zona,
                      logos_dir=a.logos_dir, top_n=a.top, historias=not a.sin_historias,
                      formatos=a.formatos)


if __name__ == "__main__":
    main()
