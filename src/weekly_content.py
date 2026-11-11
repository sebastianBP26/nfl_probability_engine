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
  3. GRÁFICAS  Las mismas imágenes en dos tamaños, con el sistema visual
               "Black Sea" (ver Estilo y FORMATOS):
                 reels/semana_XX/size_1/  1080x902   (publicación)
                 reels/semana_XX/size_2/  1080x1920  (reel / historia)
               Cada carpeta lleva:
                 01  aciertos por semana + acumulado
                 02  quién sube / quién cae en ELO (la semana pasada)
                 03  quién sube / quién cae en ELO (desde la pretemporada)
                 04  top 10 según el ELO (tabla con cambio de lugar)
                 XX  picks de la semana, en orden de juego y con escudos
                     (XX = semana - 1: semana 6 -> 05_picks_semana_06)
                 06  el más claro
                 07  la moneda al aire
                 08  la sorpresa
                 09  evolución del ELO de la temporada (top 5 en color)
                 10  evolución del ELO por división (los 32 equipos)
                 partidos/  una historia por partido (récord, escudos,
                            probabilidad y comentario) + comentarios.md
               + resumen.md con los números para el guion y las tablas en CSV
               (en la raíz de reels/semana_XX/).
  4. MODELOS   Corre también el Modelo 2 (Elo + ausencias, src/elo_ausencias.py),
               congela los pronósticos de AMBOS modelos en el libro
               data/processed/pronosticos/pronosticos_<season>.csv y agrega a
               resumen.md la sección "Modelo 1 vs Modelo 2" (+ comparacion_modelos.csv
               y ausencias_semana_XX.csv). Las imágenes de la raíz son del Modelo 1;
               el set completo del Modelo 2 va en reels/semana_XX/modelo_2/.

Uso (desde la raíz del repo):
    python src/weekly_content.py                    # semana automática (martes)
    python src/weekly_content.py --solo-pronosticos # viernes/sábado: re-congela el Modelo 2
                                                    # con el reporte final de lesiones
    python src/weekly_content.py --sin-ausencias    # solo el Modelo 1, como antes
    python src/weekly_content.py --formatos size_1  # solo las imágenes 1080x902
    python src/weekly_content.py --threshold 0.6    # solo cuenta picks con > 60 %
    python src/weekly_content.py --semana 5         # rehacer el contenido de la semana 5
    python src/weekly_content.py --sin-actualizar   # no reescribe data/processed ni el libro

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
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter
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
from src import model_registry as libro_pron  # noqa: E402

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
# Semana desde la que el modelo PUBLICADO es el Modelo 2 (Elo + ausencias). La
# calibración de semana_XX/modelo_2/ junta: Modelo 1 antes de esa semana y
# Modelo 2 desde ella (lo congelado en el libro de pronósticos).
CAMBIO_MODELO = {2026: 10}

DATA_DIR = ROOT / "data" / "processed"
LOGOS_DIR = ROOT / "data" / "logos"      # caché local de escudos
REELS_DIR = ROOT / "reels"               # una carpeta por semana

# ══════════════════════════════════════════════════════════════════════════════
# Estilo — sistema visual "Black Sea" (diseño plano: sin degradados, sombras,
# texturas ni cuadrícula de fondo)
# ══════════════════════════════════════════════════════════════════════════════
FONDO = "#332E32"        # Black Sea: siempre liso
PANEL = "#3E383D"        # paneles, pistas de barras y filas resaltadas
LINEA = "#554E54"        # líneas divisorias y segmentos secundarios
NEUTRO = "#686167"       # segmento neutro de las barras
TEXTO = "#F6F2EE"        # texto principal (blanco cálido)
TEXTO2 = "#C2BBC0"       # texto secundario: etiquetas, valores de apoyo
TENUE = "#8E878C"        # notas al pie, encabezados de columna
AZUL = "#7778DE"         # Blueberry Pie (relleno): favorito / positivo / lo que sube
AZUL_TXT = "#8F90EC"     # Blueberry como texto sobre el fondo
ROJO = "#F44631"         # Cherry Tomato (relleno): protagonista / sorpresa / lo que cae
ROJO_TXT = "#F65A47"     # Cherry Tomato como texto sobre el fondo
BLANCO = "#FFFFFF"       # texto encima de un relleno (siempre en negritas)

W, H, DPI = 9, 16, 120   # 1080 x 1920 — el lienzo de diseño (ver FORMATOS)
ANCHO_PX, ALTO_PX = W * DPI, H * DPI
MARGEN = 64              # márgenes laterales (px)
FUENTE_MIN = 11          # px: ningún texto baja de aquí al escalar a 1080x902
# Zona segura de la historia de IG: el contenido va entre y = 300 y y = 1780 px
Y_ETIQUETA, Y_TITULO, Y_SUBTITULO, Y_LINEA, Y_PIE = 300, 352, 436, 494, 1650

ESTILO = {
    "font.family": "DejaVu Sans",
    "text.color": TEXTO,
    "axes.labelcolor": TEXTO,
    "xtick.color": TEXTO2,
    "ytick.color": TENUE,
    "axes.unicode_minus": False,
}

DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]
ZONAS_CORTAS = {"America/Mexico_City": "CDMX", "America/New_York": "ET",
                "America/Chicago": "CT", "America/Los_Angeles": "PT",
                "Europe/Madrid": "Madrid"}


DECIMALES = 0   # porcentajes enteros en todo lo que se publica
RANKING_N = 10  # equipos en la tabla del ranking (04_top_elo)


def fmt_pct(p, d=DECIMALES):
    """0.5012 -> '50%'"""
    return f"{p:.{d}%}"


def fmt_par(p_a, p_b, d=DECIMALES):
    """
    Formatea dos probabilidades complementarias para que siempre sumen 100.
    Redondea la primera y la segunda es 100 menos la primera, para evitar
    casos como 51% vs 50%.
    """
    a = round(p_a * 100, d)
    return f"{a:.{d}f}%", f"{100 - a:.{d}f}%"


def par_enteros(p_a):
    """Probabilidad y su complemento como enteros que suman exactamente 100."""
    a = int(round(float(p_a) * 100))
    return a, 100 - a


def par_decimal(p_a, d=1):
    """Como par_enteros pero con d decimales (50.3 / 49.7): así un 50-50 deja ver al favorito."""
    a = round(float(p_a) * 100, d)
    return a, round(100 - a, d)
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

    def _hora_ok(gametime):
        return isinstance(gametime, str) and ":" in gametime

    def _kickoff(gameday, gametime):
        if pd.isna(gameday):
            return pd.NaT
        hora = gametime if _hora_ok(gametime) else "13:00"
        dt = datetime.strptime(f"{str(gameday)[:10]} {hora[:5]}", "%Y-%m-%d %H:%M")
        return pd.Timestamp(dt.replace(tzinfo=et).astimezone(local))

    g["kickoff"] = [_kickoff(d, t) for d, t in zip(g["gameday"], g["gametime"])]
    g["hora_confirmada"] = [_hora_ok(t) for t in g["gametime"]]   # si no, "por confirmar"

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

    El más claro       mayor probabilidad del favorito.
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
    out = [("El más claro", r,
            f"El favorito más claro de la semana:\n{r.favorito} gana con {fmt_pct(r.p_fav)}")]

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
            # versión para fondo oscuro (la misma ruta de ESPN con /500-dark/)
            "espn_dark": {t: u.replace("/500/", "/500-dark/") if isinstance(u, str) else u
                          for t, u in zip(teams_desc["team_abbr"], teams_desc["team_logo_espn"])},
            # respaldo: escudo cuadrado de nflverse (fondo sólido, se recorta en círculo)
            "squared": dict(zip(teams_desc["team_abbr"], teams_desc["team_logo_squared"])),
            "division": dict(zip(teams_desc["team_abbr"], teams_desc["team_division"])),
        }
    return _logo_urls


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
    carpeta = (LOGOS_DIR if fuente == "espn" else
               LOGOS_DIR / "_dark" if fuente == "espn_dark" else LOGOS_DIR / f"_{fuente}")
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
    for fuente, preparar in (("espn_dark", _recortar_transparente),
                             ("espn", _recortar_transparente), ("squared", _circulo)):
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
    Tamaño de la imagen. Todas las gráficas se diseñan en pixeles del lienzo
    vertical 1080x1920 (x desde la izquierda, y desde arriba); la franja
    y0 → y1 de ese diseño (fracciones desde abajo) se lleva a dest0 → dest1 del
    lienzo real, y fuentes / escudos se escalan en la misma proporción.
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
        """Factor en pixeles: fuentes, escudos, íconos."""
        return self.k * self.alto / H


FORMATOS = {
    # 1080x902 (panel de video): la franja del vertical que va del encabezado
    # (y = 262 px) al pie (y = 1670 px), escalada para que quepa
    "size_1": Formato(W, 902 / DPI, y0=1 - 1670 / ALTO_PX, y1=1 - 262 / ALTO_PX,
                      dest0=0.03, dest1=0.97),
    # 1080x1920 (historia de IG): el diseño original, sin cambios
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
    """y del diseño (fracción desde abajo) -> y del lienzo actual (fracción de la figura)."""
    return _FMT.dest0 + (v - _FMT.y0) * _FMT.k


def _X(px):
    """x en px del diseño -> fracción de la figura (el ancho no se escala)."""
    return px / ANCHO_PX


def _Y(px):
    """y en px del diseño (desde arriba) -> fracción de la figura actual."""
    return _y(1 - px / ALTO_PX)


def _fs(n):
    """Tamaño de marcador en puntos escalado al formato."""
    return n * _FMT.escala


def _ft(px, minimo=FUENTE_MIN):
    """Fuente en px del lienzo de 1080 de ancho -> puntos, escalada al formato."""
    return max(px * _FMT.escala, minimo) * 72 / DPI


def _lw(px):
    """Grosor de línea en px -> puntos (las líneas finas no se escalan)."""
    return px * 72 / DPI


def _e(px):
    """Distancia horizontal que crece con la fuente (íconos, píldoras)."""
    return px * _FMT.escala


def _px(n):
    """Tamaño de escudo en pixeles escalado al formato."""
    return max(1, int(round(n * _FMT.escala)))


def _ancho_texto(s, px, bold=True):
    """Ancho real (px) de un texto con DejaVu Sans a px pixeles."""
    prop = FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return TextPath((0, 0), s, size=px, prop=prop).get_extents().width


def _capa(fig):
    """Ejes en pixeles reales del lienzo, para rectángulos, círculos y líneas."""
    ax = getattr(fig, "_capa_px", None)
    if ax is None:
        w, h = fig.get_size_inches() * fig.dpi
        ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.axis("off")
        fig._capa_px = ax
    return ax


def _a_px(fig, x, y):
    """(x, y) del diseño -> pixeles reales del lienzo (origen abajo a la izquierda)."""
    w, h = fig.get_size_inches() * fig.dpi
    return _X(x) * w, _Y(y) * h


def _caja(fig, x0, y0, x1, y1, color, radio=0):
    """Rectángulo plano de (x0, y0) a (x1, y1) en px del diseño; radio en px."""
    (X0, Ya), (X1, Yb) = _a_px(fig, x0, y1), _a_px(fig, x1, y0)
    if radio:
        p = FancyBboxPatch((X0, Ya), X1 - X0, Yb - Ya, facecolor=color, edgecolor="none",
                           boxstyle=f"round,pad=0,rounding_size={_e(radio)}")
    else:
        p = Rectangle((X0, Ya), X1 - X0, Yb - Ya, facecolor=color, edgecolor="none")
    _capa(fig).add_patch(p)


def _hline(fig, x0, x1, y, color=LINEA, grosor=1):
    (X0, Y), (X1, _) = _a_px(fig, x0, y), _a_px(fig, x1, y)
    _capa(fig).plot([X0, X1], [Y, Y], color=color, lw=_lw(grosor), solid_capstyle="butt")


def _vline(fig, x, y0, y1, color=LINEA, grosor=1):
    (X, Y0), (_, Y1) = _a_px(fig, x, y0), _a_px(fig, x, y1)
    _capa(fig).plot([X, X], [Y0, Y1], color=color, lw=_lw(grosor), solid_capstyle="butt")


def _texto(fig, x, y, s, px, color=TEXTO, bold=False, ha="left", va="center",
           minimo=FUENTE_MIN, **kw):
    """Texto en (x, y) px del diseño, con tamaño en px."""
    return fig.text(_X(x), _Y(y), s, fontsize=_ft(px, minimo), color=color,
                    fontweight="bold" if bold else "normal", ha=ha, va=va, **kw)


def _icono(fig, cx, cy, glifo):
    """Ícono de sección: círculo Blueberry de 68 px con un glifo blanco en negritas."""
    X, Y = _a_px(fig, cx, cy)
    _capa(fig).add_patch(Circle((X, Y), _e(34), facecolor=AZUL, edgecolor="none"))
    _texto(fig, cx, cy, glifo, 34, BLANCO, bold=True, ha="center")


def _seccion(fig, y, glifo, titulo, pregunta):
    """Ícono + título de sección (30 px) + subtítulo en forma de pregunta (20 px)."""
    _icono(fig, MARGEN + _e(34), y, glifo)
    x = MARGEN + _e(90)
    _texto(fig, x, y - 14, titulo, 30, TEXTO, bold=True)
    _texto(fig, x, y + 21, pregunta, 20, AZUL_TXT)


def _pildora(fig, cx, y0, cifra, leyenda, ancho=540, alto=150):
    """Píldora Cherry Tomato (radio 12): cifra grande en blanco + leyenda de 13 px."""
    w = _e(ancho) / 2
    _caja(fig, cx - w, y0, cx + w, y0 + alto, ROJO, radio=12)
    _texto(fig, cx, y0 + alto * 0.44, cifra, 80, BLANCO, bold=True, ha="center")
    _texto(fig, cx, y0 + alto * 0.83, leyenda, 13, BLANCO, bold=True, ha="center")


def _barra(fig, x0, y0, x1, y1, segmentos, px, decimales=0):
    """
    Barra de probabilidad: segmentos [(valor, color)] que suman 100, con 2 px
    de separación y el porcentaje en negritas centrado dentro de cada uno.
    """
    seg = [(v, c) for v, c in segmentos if v > 0]
    total = (x1 - x0) - 2 * (len(seg) - 1)
    x = x0
    for v, color in seg:
        w = total * v / 100
        _caja(fig, x, y0, x + w, y1, color)
        etiqueta, tam = f"{v:.{decimales}f}%", px
        while tam > px * 0.6 and _ancho_texto(etiqueta, tam * _FMT.escala) > w - 10:
            tam -= 1
        if _ancho_texto(etiqueta, tam * _FMT.escala) <= w - 6:
            _texto(fig, x + w / 2, (y0 + y1) / 2, etiqueta, tam, BLANCO, bold=True, ha="center")
        x += w + 2


_MODELO_PIE = None   # con usar_modelo("Elo + ausencias") el pie dice de qué modelo es


@contextmanager
def usar_modelo(nombre):
    """Marca en el pie de cada imagen el modelo que la generó (None = Modelo 1, sin marca)."""
    global _MODELO_PIE
    previo, _MODELO_PIE = _MODELO_PIE, nombre
    try:
        yield
    finally:
        _MODELO_PIE = previo


def _lienzo(etiqueta, titulo, subtitulo=None, tema="Modelo Elo"):
    """
    Fondo liso + encabezado (etiqueta Cherry, título, subtítulo, línea de 3 px)
    + pie ("— TEMA · No es consejo de apuesta" y @samplesebs).
    """
    if _MODELO_PIE:
        tema = f"{_MODELO_PIE} · {tema}"
    fig = plt.figure(figsize=(_FMT.ancho, _FMT.alto), dpi=DPI, facecolor=FONDO)
    _capa(fig)
    _texto(fig, 540, Y_ETIQUETA, etiqueta.upper(), 33, ROJO_TXT, bold=True, ha="center", va="top")
    _texto(fig, 540, Y_TITULO, titulo, 56, TEXTO, bold=True, ha="center", va="top")
    if subtitulo:
        _texto(fig, 540, Y_SUBTITULO, subtitulo, 25, TEXTO2, ha="center", va="top")
    _hline(fig, MARGEN, ANCHO_PX - MARGEN, Y_LINEA, ROJO, 3)
    _texto(fig, MARGEN, Y_PIE, f"— NFL · {tema.upper()} · No es consejo de apuesta", 16, TENUE,
           minimo=12)
    _texto(fig, ANCHO_PX - MARGEN, Y_PIE, "@samplesebs", 16, TEXTO2, bold=True, ha="right",
           minimo=12)
    return fig


def _guardar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=DPI, facecolor=FONDO)
    plt.close(fig)
    return ruta


def _logo(fig, team, x, y, px):
    """
    Escudo centrado en (x, y) px del diseño con tamaño óptico px. Si no hay
    escudo: círculo #3E383D con borde #554E54 y las iniciales en #C2BBC0.
    """
    px = _px(px)
    img = obtener_logo(team, px)
    fig_w, fig_h = fig.get_size_inches() * fig.dpi
    fx, fy = _X(x), _Y(y)
    if img is None:
        _capa(fig).add_patch(Circle((fx * fig_w, fy * fig_h), px / 2, facecolor=PANEL,
                                    edgecolor=LINEA, lw=_lw(2)))
        fig.text(fx, fy, team, ha="center", va="center", fontsize=px * 0.3 * 72 / DPI,
                 fontweight="bold", color=TEXTO2)
        return None
    alto_px, ancho_px = img.shape[:2]
    w, h = ancho_px / fig_w, alto_px / fig_h
    ax = fig.add_axes([fx - w / 2, fy - h / 2, w, h])
    ax.axis("off")
    ax.imshow(img)
    return ax


def _ejes(fig, x0, y0, x1, y1):
    """Ejes de matplotlib que ocupan de (x0, y0) a (x1, y1) px del diseño."""
    return fig.add_axes([_X(x0), _Y(y1), _X(x1) - _X(x0), _Y(y0) - _Y(y1)], facecolor="none")


def _fig_y(ax, y):
    """Coordenada y de datos -> coordenada y de la figura."""
    pos = ax.get_position()
    y0, y1 = ax.get_ylim()
    return pos.y0 + pos.height * (y - y0) / (y1 - y0)


def _dy(ax, v):
    """Coordenada y de datos -> y en px del diseño (desde arriba)."""
    f = _fig_y(ax, v)
    return (1 - ((f - _FMT.dest0) / _FMT.k + _FMT.y0)) * ALTO_PX


def _sin_bordes(ax, dejar=()):
    for lado, s in ax.spines.items():
        s.set_visible(lado in dejar)


def _eje_x(ax, etiquetas, px=21, color=TEXTO2, **kw):
    """Etiquetas del eje x sin marcas, con línea base fina."""
    ax.set_xticks(np.arange(len(etiquetas)))
    ax.set_xticklabels(etiquetas, fontsize=_ft(px), color=color, **kw)
    ax.tick_params(axis="x", length=0, pad=_fs(8))
    ax.set_yticks([])
    _sin_bordes(ax, dejar=("bottom",))
    ax.spines["bottom"].set_color(LINEA)
    ax.spines["bottom"].set_linewidth(_lw(2))


def _zona_corta(zona):
    return ZONAS_CORTAS.get(zona, zona.split("/")[-1].replace("_", " "))


def _fmt_dia(ts):
    """'jue 5 nov'"""
    return f"{DIAS[ts.weekday()]} {ts.day} {MESES[ts.month - 1]}"


def _fmt_fecha_hora(ts, zona, confirmada=True):
    """'jue 5 nov · 19:15 · hora CDMX' — sin hora si no está confirmada."""
    if pd.isna(ts):
        return "Fecha por confirmar"
    if not confirmada:
        return f"{_fmt_dia(ts)} · hora por confirmar"
    return f"{_fmt_dia(ts)} · {ts:%H:%M} · hora {_zona_corta(zona)}"


def _miles(v):
    """1578.3 -> '1,578'"""
    return f"{v:,.0f}"


def _separar(ys, gap):
    """Reparte posiciones (1-D) para que queden al menos `gap` entre sí, sin perder el orden."""
    orden = np.argsort(ys)[::-1]                 # de mayor a menor
    out = np.asarray(ys, dtype=float).copy()
    for a, b in zip(orden[:-1], orden[1:]):
        out[b] = min(out[b], out[a] - gap)
    exceso = out.min() - min(ys)                 # si se fue muy lejos, centrar el bloque
    if exceso < 0:
        out -= exceso / 2
    return out


def _separar_px(ys_px, gap):
    """_separar para px del diseño (y crece hacia abajo)."""
    return -_separar(-np.asarray(ys_px, dtype=float), gap)


# ══════════════════════════════════════════════════════════════════════════════
# 6 · Gráficas
# ══════════════════════════════════════════════════════════════════════════════
def graf_aciertos_semana(tabla, threshold, ruta):
    """Récord acumulado en píldora · barras = % de aciertos de cada semana ·
    línea = % acumulado. Los conteos van como récord de la NFL."""
    t = tabla.reset_index(drop=True)
    u = t.iloc[-1]
    fig = _lienzo("La prueba", "¿Cómo va el modelo?",
                  f"{u.acum_pct_aciertos:.0%} de aciertos en {int(u.acum_picks)} partidos "
                  f"· semanas 1–{int(u.semana)}", tema="Desempeño del modelo")

    leyenda = ("RÉCORD DEL MODELO · ACIERTOS–FALLOS–EMPATES" if u.acum_empates
               else "RÉCORD DEL MODELO · ACIERTOS–FALLOS")
    _pildora(fig, 540, 532, record_af(u.acum_aciertos, u.acum_picks, u.acum_empates), leyenda)
    if threshold > 0.5:
        _texto(fig, 540, 712, f"Solo cuentan los partidos con probabilidad > {threshold:.0%}",
               15, TENUE, ha="center")

    _seccion(fig, 800, "#", "Semana a semana", "¿Le gana a la moneda al aire?")
    ax = _ejes(fig, 108, 880, 972, 1280)
    n = len(t)
    x = np.arange(n)
    pct = t["pct_aciertos"].fillna(0).to_numpy()
    # semana con 50 % o más: Blueberry · semana perdedora: Cherry Tomato
    ax.bar(x, pct, width=0.62, color=[AZUL if p >= 0.5 else ROJO for p in pct], zorder=2)

    vertical = n > 9
    fs = 24 if n <= 6 else (21 if n <= 9 else 18)
    for xi, r in zip(x, t.itertuples()):
        ax.text(xi, 0.03, record_af(r.aciertos, r.picks, r.empates), ha="center", va="bottom",
                fontsize=_ft(fs), fontweight="bold", color=BLANCO,
                rotation=90 if vertical else 0, zorder=5)

    acum = t["acum_pct_aciertos"].to_numpy()
    ax.plot(x, acum, color=TEXTO, lw=_lw(5), marker="o", ms=_fs(12), zorder=6,
            markeredgecolor=FONDO, markeredgewidth=_fs(2.5))
    ax.annotate(f"{acum[-1]:.0%}", (x[-1], acum[-1]), xytext=(0, _fs(14)),
                textcoords="offset points", ha="center", fontsize=_ft(32),
                fontweight="bold", color=TEXTO, zorder=7)

    pad = max(0.0, (6 - n) / 2)
    ax.set_xlim(-0.6 - pad, n - 0.4 + pad)
    ax.set_ylim(0, min(1.15, max(0.8, float(np.nanmax(np.r_[pct, acum])) + 0.2)))
    ax.axhline(0.5, color=TENUE, lw=_lw(2), ls=(0, (4, 3)), zorder=3)
    fig.text(_X(100), _fig_y(ax, 0.5), "50%", ha="right", va="center", fontsize=_ft(15),
             color=TENUE)
    _eje_x(ax, [f"S{int(w)}" for w in t["semana"]], px=21 if n <= 10 else 16)

    _texto(fig, 520, 1372, "■ récord de la semana", 21, AZUL_TXT, ha="right")
    _texto(fig, 560, 1372, "● % acumulado", 21, TEXTO)
    _texto(fig, 540, 1410, "- - -  50% = moneda al aire", 15, TENUE, ha="center")

    comp = ["Siempre el local: "
            f"{record_af(u.acum_siempre_local_aciertos, u.acum_picks, u.acum_empates)}"]
    if MOSTRAR_VEGAS and u.acum_vegas_partidos:
        comp.append("Favorito de Las Vegas: "
                    f"{record_af(u.acum_vegas_aciertos, u.acum_vegas_partidos, u.acum_vegas_empates)}")
    _texto(fig, 540, 1470, "   ·   ".join(comp), 21, TEXTO2, ha="center")
    return _guardar(fig, ruta)


def graf_calibracion(calib, ruta, alta=0.70, nota=None):
    """Barras = % real que ganó el favorito en cada rango de confianza ·
    raya = lo que el modelo prometió (prob. media). Arriba, confiar vs no confiar.
    nota: (titular, detalle) opcional abajo de la leyenda, p. ej. para avisar
    de un cambio de modelo."""
    c = calib[calib["partidos"] > 0].reset_index(drop=True)
    fig = _lienzo("La prueba", "¿Cuándo hay que creerle?",
                  "Si el modelo dice 70%, ¿el favorito gana 70% de las veces?",
                  tema="Calibración del modelo")

    def _resumen(filas):
        n, a, e = filas["partidos"].sum(), filas["aciertos"].sum(), filas["empates"].sum()
        return (a + 0.5 * e) / n if n else np.nan, record_af(a, n, e)

    lim = [float(r.split("–")[0].rstrip("%+")) / 100 for r in c["rango"].astype(str)]
    conf = c[[v >= alta for v in lim]]
    dudas = c.iloc[:1]
    bloques = ((290, conf, f"≥ {alta:.0%}", AZUL_TXT),
               (790, dudas, str(dudas["rango"].iloc[0]) if len(dudas) else "", TEXTO))
    for cx, filas, rango, color in bloques:
        pct, rec = _resumen(filas)
        if np.isnan(pct):
            continue
        _texto(fig, cx, 590, f"{pct:.0%}", 80, color if pct >= 0.5 else ROJO_TXT, bold=True,
               ha="center")
        _texto(fig, cx, 672, f"ACIERTA CUANDO DA\n{rango}", 16, TENUE, bold=True, ha="center",
               linespacing=1.5)
        _texto(fig, cx, 728, rec, 21, TEXTO2, ha="center")
    _vline(fig, 540, 548, 740)

    _seccion(fig, 808, "%", "Por nivel de confianza", "¿Cumple lo que promete?")
    ax = _ejes(fig, 108, 890, 972, 1290)
    n = len(c)
    x = np.arange(n)
    real = c["pct_real"].to_numpy()
    prom = c["prob_media"].to_numpy()
    colores = [ROJO if r < 0.5 else (AZUL if r >= p - 0.02 else NEUTRO)
               for r, p in zip(real, prom)]
    ax.bar(x, real, width=0.62, color=colores, zorder=2)
    for xi, r, p, rec in zip(x, real, prom, c["record"]):
        ax.text(xi, 0.03, rec, ha="center", va="bottom", fontsize=_ft(21),
                fontweight="bold", color=BLANCO, zorder=5)
        ax.text(xi, max(r, p) + 0.03, f"{r:.0%}", ha="center", va="bottom", fontsize=_ft(28),
                fontweight="bold", color=ROJO_TXT if r < 0.5 else TEXTO, zorder=7)
        ax.plot([xi - 0.36, xi + 0.36], [p, p], color=TEXTO, lw=_lw(6),
                solid_capstyle="butt", zorder=6)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, color=TENUE, lw=_lw(2), ls=(0, (4, 3)), zorder=3)
    fig.text(_X(100), _fig_y(ax, 0.5), "50%", ha="right", va="center", fontsize=_ft(15),
             color=TENUE)
    _eje_x(ax, [f"{r}\n{int(p)} partidos" for r, p in zip(c["rango"], c["partidos"])],
           linespacing=1.5)

    _texto(fig, 520, 1418, "■ lo que acertó", 21, AZUL_TXT, ha="right")
    _texto(fig, 560, 1418, "▬ lo que prometió", 21, TEXTO)
    _texto(fig, 540, 1456, "- - -  50% = moneda al aire", 15, TENUE, ha="center")
    if nota:
        titular, detalle = nota
        _hline(fig, MARGEN, ANCHO_PX - MARGEN, 1502)
        _texto(fig, 540, 1530, titular, 22, AZUL_TXT, bold=True, ha="center", va="top")
        _texto(fig, 540, 1572, detalle, 18, TEXTO2, ha="center", va="top")
    return _guardar(fig, ruta)


def nota_cambio_modelo(semana, cambio):
    """Texto de la gráfica de calibración del historial publicado (Modelo 1 -> 2)."""
    titular = ("Nuevo: el modelo ahora toma en cuenta lesiones y ausencias" if semana <= cambio + 1
               else "El modelo toma en cuenta lesiones y ausencias")
    if semana <= cambio:
        detalle = f"Semanas 1–{cambio - 1}: modelo Elo · desde la semana {cambio}: Elo + ausencias"
    elif semana == cambio + 1:
        detalle = f"Semanas 1–{cambio - 1}: modelo Elo · semana {cambio}: Elo + ausencias"
    else:
        detalle = (f"Semanas 1–{cambio - 1}: modelo Elo · semanas {cambio}–{semana - 1}: "
                   "Elo + ausencias")
    return titular, detalle


def graf_suben_caen(cambios, contexto, etiqueta, titulo, subtitulo, ruta, n=TOP_N):
    """Top n que más Elo ganan (Blueberry) y top n que más pierden (Cherry Tomato)."""
    df = cambios.dropna().sort_values(ascending=False)
    grupos = (("SUBEN", df.head(n)), ("CAEN", df.tail(n)))
    fig = _lienzo(etiqueta, titulo, subtitulo, tema="Cambios de Elo")

    lim = max(float(df.abs().max()), 1.0) * 1.6
    x0b, x1b = 330, ANCHO_PX - MARGEN
    cero, media = (x0b + x1b) / 2, (x1b - x0b) / 2
    FILA, ALTO = 82, 46
    y = 540
    for i, (nombre, sel) in enumerate(grupos):
        if i:
            _hline(fig, MARGEN, ANCHO_PX - MARGEN, y - 22)
        _texto(fig, MARGEN, y, nombre, 16, TENUE, bold=True)
        y += 22
        _vline(fig, cero, y + 6, y + FILA * len(sel) - 6, LINEA, 2)
        for team, c in sel.items():
            yc = y + FILA / 2
            _logo(fig, team, 100, yc, 64)
            _texto(fig, 162, yc, team, 28)
            w = media * abs(c) / lim
            der = c >= 0
            color = AZUL if c > 0 else (ROJO if c < 0 else NEUTRO)
            _caja(fig, cero if der else cero - w, yc - ALTO / 2, cero + w if der else cero,
                  yc + ALTO / 2, color)
            _texto(fig, cero + (w + 12 if der else -(w + 12)), yc, f"{c:+.0f}", 28, TEXTO,
                   bold=True, ha="left" if der else "right")
            if contexto:   # resultado/récord del lado contrario a la barra
                _texto(fig, cero + (-14 if der else 14), yc, contexto.get(team, "Descansó"), 21,
                       TEXTO2, ha="right" if der else "left")
            y += FILA
        y += 44
    return _guardar(fig, ruta)


def graf_top(elo_actual, ruta, n=RANKING_N, elo_previo=None, semana=None):
    """Ranking del modelo en tabla: posición (y cambio vs la semana pasada),
    escudo, barra de Elo y valor. La fila líder va resaltada."""
    df = elo_actual.sort_values(ascending=False).head(n)
    rank_now = elo_actual.rank(ascending=False, method="min")
    rank_prev = (elo_previo.rank(ascending=False, method="min")
                 if elo_previo is not None else None)
    fig = _lienzo("Ranking del modelo", f"Top {n} según el Elo",
                  f"Elo al arrancar la semana {semana}" if semana else None, tema="Ranking Elo")

    FILA = 56
    alto = 40 + n * FILA + 70
    top = Y_LINEA + max(40, (1600 - Y_LINEA - alto) / 2)
    der = ANCHO_PX - MARGEN
    _texto(fig, 100, top + 12, "#", 16, TENUE, bold=True, ha="center")
    _texto(fig, 182, top + 12, "EQUIPO", 16, TENUE, bold=True)
    _texto(fig, der, top + 12, "ELO", 16, TENUE, bold=True, ha="right")
    y = top + 36
    _hline(fig, MARGEN, der, y)

    base = 1400
    tope = max(float(df.max()), base + 1)
    bx0, bx1 = 340, 880
    for i, (team, v) in enumerate(df.items()):
        lider = i == 0
        yc = y + FILA / 2
        if lider:
            _caja(fig, MARGEN, y + 1, der, y + FILA, PANEL)
        pos = int(rank_now[team])
        _texto(fig, 100, yc, f"{pos}", 28, ROJO_TXT if lider else TEXTO2, bold=True, ha="center")
        if rank_prev is not None:
            d = int(rank_prev[team]) - pos
            s, col = ((f"▲{d}", AZUL_TXT) if d > 0 else
                      (f"▼{-d}", ROJO_TXT) if d < 0 else ("=", TENUE))
            _texto(fig, 100 + _e(24), yc, s, 11, col, bold=True)
        _logo(fig, team, 200, yc, 36)
        _texto(fig, 236, yc, team, 28, TEXTO, bold=lider)
        _caja(fig, bx0, yc - 12, bx0 + (bx1 - bx0) * max(v - base, 0) / (tope - base), yc + 12,
              AZUL)
        _texto(fig, der, yc, _miles(v), 28, ROJO_TXT if lider else TEXTO, bold=True, ha="right")
        y += FILA
        _hline(fig, MARGEN, der, y)

    nota = "Promedio de la liga ≈ 1,500"
    if rank_prev is not None:
        nota += " · ▲▼ lugares que subió o bajó vs la semana pasada"
    _texto(fig, 540, y + 44, nota, 15, TENUE, ha="center")
    return _guardar(fig, ruta)


def graf_evolucion_elo(elo_tab, semana, ruta, n=TOP_N):
    """Elo de los 32 equipos semana a semana desde la pretemporada.
    Todos en gris; el top n en Blueberry y el que más cae en Cherry Tomato."""
    cols = [c for c in elo_tab.columns if c <= semana]
    t = elo_tab[cols].dropna()
    actual, inicio = t[cols[-1]], t[cols[0]]
    top = list(actual.sort_values(ascending=False).index[:n])
    cae = (actual - inicio).idxmin()
    destacados = top + ([cae] if cae not in top else [])

    fig = _lienzo("Evolución del Elo", "Así se mueve la liga",
                  f"De la pretemporada a la semana {semana - 1}", tema="Evolución del Elo")
    ax = _ejes(fig, 160, 560, 800, 1440)
    x = np.arange(len(cols))
    for team, fila in t.iterrows():
        if team not in destacados:
            ax.plot(x, fila.to_numpy(), color=LINEA, lw=_lw(2), zorder=2)
    for z, team in enumerate(destacados[::-1]):    # el #1 queda encima
        color = ROJO if team == cae and team not in top else AZUL
        y = t.loc[team].to_numpy()
        ax.plot(x, y, color=color, lw=_lw(5), zorder=4 + z, solid_capstyle="round")
        ax.plot(x[-1], y[-1], "o", ms=_fs(11), color=color, zorder=4 + z,
                markeredgecolor=FONDO, markeredgewidth=_fs(2))

    lo, hi = float(t.min().min()), float(t.max().max())
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.25, len(cols) - 0.75)
    ax.axhline(1500, color=TENUE, lw=_lw(2), ls=(0, (4, 3)), zorder=3)
    ax.text(0.02, 1500, "promedio ≈ 1,500", transform=ax.get_yaxis_transform(),
            va="center", fontsize=_ft(15), color=TENUE, zorder=10,
            bbox=dict(boxstyle="square,pad=0.3", facecolor=FONDO, edgecolor="none"))

    _eje_x(ax, ["Pre"] + [f"S{c - 1}" for c in cols[1:]], px=21 if len(cols) <= 10 else 15)
    paso = 50 if hi - lo < 300 else 100
    ax.set_yticks(np.arange(np.ceil((lo - pad) / paso) * paso, hi + pad, paso))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _miles(v)))
    ax.tick_params(axis="y", length=0, labelsize=_ft(15), labelcolor=TENUE)

    # Escudo + Elo al final de cada línea destacada, sin encimarse
    ys = _separar_px([_dy(ax, actual[team]) for team in destacados], 46)
    for i, (team, y) in enumerate(zip(destacados, ys)):
        _logo(fig, team, 842, y, 36)
        lider, es_cae = i == 0, team == cae and team not in top
        _texto(fig, 872, y, _miles(actual[team]), 21,
               ROJO_TXT if es_cae else (TEXTO if lider else TEXTO2), bold=lider or es_cae)

    _texto(fig, 540, 1512, f"En azul: top {n} · en rojo: el que más cae · en gris: el resto",
           15, TENUE, ha="center")
    return _guardar(fig, ruta)


DIVISIONES = {"East": "Este", "North": "Norte", "South": "Sur", "West": "Oeste"}
# color de línea / de texto según el lugar en la división (1.º → 4.º)
COLORES_DIVISION = [(AZUL, AZUL_TXT), (TEXTO, TEXTO), (TENUE, TEXTO2), (ROJO, ROJO_TXT)]


def graf_evolucion_divisiones(elo_tab, semana, ruta):
    """Elo semana a semana en 8 paneles (AFC a la izquierda, NFC a la derecha),
    los 4 equipos de cada división por lugar (1.º Blueberry → 4.º Cherry) y
    con la misma escala en todos."""
    cols = [c for c in elo_tab.columns if c <= semana]
    t = elo_tab[cols].dropna()
    division = _load_logo_urls()["division"]
    lo, hi = float(t.min().min()), float(t.max().max())
    pad = (hi - lo) * 0.08
    x = np.arange(len(cols))
    etiquetas = ["Pre"] + [f"S{c - 1}" for c in cols[1:]]
    if len(cols) > 10:   # con muchas semanas, una etiqueta sí y otra no
        etiquetas = [e if i % 2 == 0 else "" for i, e in enumerate(etiquetas)]

    fig = _lienzo("Evolución del Elo", "Así va cada división",
                  f"De la pretemporada a la semana {semana - 1}", tema="Evolución del Elo")
    FILA, top = 250, 524
    for col, conf in enumerate(("AFC", "NFC")):
        x0 = MARGEN + 488 * col
        for fila, (div_en, div_es) in enumerate(DIVISIONES.items()):
            ry = top + fila * FILA
            _texto(fig, x0, ry, f"{conf} {div_es}".upper(), 16, TEXTO2, bold=True, va="top")
            equipos = [e for e in t.index if division.get(e) == f"{conf} {div_en}"]
            equipos.sort(key=lambda e: -t.loc[e, cols[-1]])

            ax = _ejes(fig, x0, ry + 30, x0 + 330, ry + FILA - 44)
            ax.set_facecolor(PANEL)
            ax.axhline(1500, color=NEUTRO, lw=_lw(1.5), ls=(0, (4, 3)), zorder=1)
            for z, (e, (c_linea, _)) in reversed(list(enumerate(zip(equipos, COLORES_DIVISION)))):
                y = t.loc[e].to_numpy()
                ax.plot(x, y, color=c_linea, lw=_lw(4), zorder=2 + (4 - z),
                        solid_capstyle="round")
                ax.plot(x[-1], y[-1], "o", ms=_fs(7), color=c_linea, zorder=2 + (4 - z),
                        markeredgecolor=PANEL, markeredgewidth=_fs(1.5))
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.4, len(cols) - 0.6)
            ultima = fila == len(DIVISIONES) - 1
            _eje_x(ax, etiquetas if ultima else [""] * len(cols), px=15, color=TENUE)
            _sin_bordes(ax)

            ys = _separar_px([_dy(ax, t.loc[e, cols[-1]]) for e in equipos], 34)
            for e, y, (_, c_txt) in zip(equipos, ys, COLORES_DIVISION):
                _logo(fig, e, x0 + 354, y, 32)
                _texto(fig, x0 + 378, y, _miles(t.loc[e, cols[-1]]), 20, c_txt, bold=True)

    _texto(fig, 540, 1572, "Misma escala en todas las divisiones · - - -  1,500 = promedio",
           15, TENUE, ha="center")
    return _guardar(fig, ruta)


def graf_picks(week_games, semana, threshold, ruta, zona=ZONA_HORARIA):
    """Pronósticos en orden de juego (jueves → lunes): escudos y barra de probabilidad.
    La fila del favorito más claro va resaltada."""
    g = week_games.sort_values(["kickoff", "game_id"], na_position="last")
    fig = _lienzo(f"Semana {semana}", "Mis pronósticos",
                  f"En orden de juego · hora {_zona_corta(zona)}",
                  tema=f"Pronósticos · semana {semana}")

    # Renglones: encabezado de día + un renglón por partido
    filas, dia_actual = [], "∅"
    for r in g.itertuples():
        dia = r.kickoff.date() if pd.notna(r.kickoff) else None
        if dia != dia_actual:
            filas.append(("dia", r))
            dia_actual = dia
        filas.append(("juego", r))
    n_dias = sum(t == "dia" for t, _ in filas)

    der = ANCHO_PX - MARGEN
    XH, XFAV, XVS, XRIV, XLOC, XB0, XB1 = MARGEN + 10, 210, 272, 335, 385, 530, der - 10
    _texto(fig, XH, 532, "HORA", 16, TENUE, bold=True)
    _texto(fig, XFAV, 532, "FAVORITO", 16, TENUE, bold=True, ha="center")
    _texto(fig, XRIV, 532, "RIVAL", 16, TENUE, bold=True, ha="center")
    _texto(fig, XB0, 532, "PROBABILIDAD DE VICTORIA", 16, TENUE, bold=True)
    _hline(fig, MARGEN, der, 552)

    H_DIA, GAP = 0.85, 0.3
    unidades = (len(filas) - n_dias) + H_DIA * n_dias + GAP * (n_dias - 1)
    top, bottom = 562, 1548
    u = min(56, (bottom - top) / unidades)
    lider = g["p_fav"].idxmax()

    y, primero, sin_pick, hora_prev = top, True, False, None
    for tipo, r in filas:
        if tipo == "dia":
            if not primero:
                y += GAP * u
            primero, hora_prev = False, None
            texto = _fmt_dia(r.kickoff).upper() if pd.notna(r.kickoff) else "FECHA POR CONFIRMAR"
            _texto(fig, MARGEN, y + H_DIA * u / 2, texto, 18, TEXTO, bold=True)
            y += H_DIA * u
            continue

        yc = y + u / 2
        es_lider = r.Index == lider
        es_pick = r.p_fav > threshold
        sin_pick |= not es_pick
        if es_lider:
            _caja(fig, MARGEN, y + 2, der, y + u - 2, PANEL)
        confirmada = pd.notna(r.kickoff) and getattr(r, "hora_confirmada", True)
        hora = f"{r.kickoff:%H:%M}" if confirmada else "por confirmar"
        if hora != hora_prev:          # la hora solo cuando cambia
            _texto(fig, XH, yc, hora, 21 if confirmada else 15, TEXTO2 if confirmada else TENUE)
            hora_prev = hora
        _logo(fig, r.favorito, XFAV, yc, 36)
        _texto(fig, XVS, yc, "vs", 16, TENUE, ha="center")
        _logo(fig, r.rival, XRIV, yc, 32)
        _texto(fig, XLOC, yc, "en casa" if r.fav_local else "de visita", 21, TEXTO2)
        # un decimal: en un 50-50 redondeado se ve igual quién es el favorito (50.3 vs 49.7)
        p_fav, p_riv = par_decimal(r.p_fav)
        # confianza alta (≥ ALTA_CONFIANZA) en Cherry Tomato, media en Blueberry
        color = (NEUTRO if not es_pick else
                 ROJO if r.p_fav >= ALTA_CONFIANZA else AZUL)
        _barra(fig, XB0, yc - u * 0.31, XB1, yc + u * 0.31, [(p_fav, color), (p_riv, LINEA)], 21,
               decimales=1)
        y += u

    r = g.loc[lider]
    p = par_decimal(r.p_fav)[0]
    # Leyenda: cuadrito de color + texto gris, centrada
    items = [(ROJO, f"Confianza alta ({ALTA_CONFIANZA:.0%} o más)"), (AZUL, "Confianza media")]
    if sin_pick:
        items.append((NEUTRO, f"Sin pronóstico (≤ {threshold:.0%})"))
    lado, sep, gap = _e(18), _e(10), _e(40)
    anchos = [lado + sep + _ancho_texto(t, 18 * _FMT.escala, bold=False) for _, t in items]
    x = 540 - (sum(anchos) + gap * (len(items) - 1)) / 2
    yl = y + 34
    for (color, t), w in zip(items, anchos):
        _caja(fig, x, yl - 9, x + lado, yl + 9, color, radio=3)
        _texto(fig, x + lado + sep, yl, t, 18, TEXTO2)
        x += w + gap
    notas = f"Aun con {p:.1f}%, {r.favorito} no gana {100 - round(p)} de cada 100 veces"
    _texto(fig, 540, y + 72, notas, 15, TENUE, ha="center")
    return _guardar(fig, ruta)


def _envolver(texto, ancho=44):
    """Parte el texto en renglones de ~ancho caracteres (respeta los \\n que ya traiga)."""
    return "\n".join(textwrap.fill(p, ancho) for p in texto.split("\n"))


BAJAS_MAX = 3   # renglones de nombres por equipo en la tarjeta de cada partido


def _baja_corta(item, largo=40):
    """'Jauan Jennings (WRTE, Out sem. 9)' -> 'Jauan Jennings (WR/TE, Out sem. 9)'."""
    t = item.replace("(WRTE,", "(WR/TE,").replace(", otro equipo)", ", ya no está)")
    return t if len(t) <= largo else t[: largo - 1].rstrip() + "…"


def _bloque_bajas(fig, r, y0):
    """
    Debajo de la tabla, alineado con cada escudo: cuánto Elo le resta el modelo
    al equipo por sus bajas y quién falta (todas las posiciones). Regresa la y
    donde termina el bloque.
    """
    _texto(fig, 540, y0, "BAJAS Y LESIONES", 16, TENUE, bold=True, ha="center")
    filas_max = 0
    for cx, bajas, aj in ((270, r.bajas_away, r.ajuste_elo_away),
                          (810, r.bajas_home, r.ajuste_elo_home)):
        lista = [b for b in str(bajas or "").split("; ") if b and b != "nan"]
        aj = float(aj or 0.0)
        if aj <= -5:
            _texto(fig, cx, y0 + 38, f"{aj:+.0f} Elo".replace("-", "−"), 24, ROJO_TXT,
                   bold=True, ha="center")
        elif lista:
            _texto(fig, cx, y0 + 38, "No mueven el pronóstico", 19, TEXTO2, ha="center")
        else:
            _texto(fig, cx, y0 + 38, "Sin bajas reportadas", 19, TEXTO2, ha="center")
        if len(lista) > BAJAS_MAX:
            lista = lista[: BAJAS_MAX - 1] + [f"+{len(lista) - BAJAS_MAX + 1} más"]
        for i, b in enumerate(lista):
            _texto(fig, cx, y0 + 72 + 28 * i, _baja_corta(b), 17, TEXTO2, ha="center")
        filas_max = max(filas_max, len(lista))
    fin = y0 + 72 + 28 * max(filas_max - 1, 0) + 22
    _hline(fig, MARGEN, ANCHO_PX - MARGEN, fin)
    return fin


def graf_duelo(r, etiqueta, explicacion, ruta, zona=ZONA_HORARIA, rec=None, rank=None,
               tema="Pronósticos"):
    """Un partido: escudos, barra de probabilidad, tabla comparativa y el comentario."""
    rec, rank = rec or {}, rank or {}
    fig = _lienzo(etiqueta, f"{r.away_team} @ {r.home_team}",
                  _fmt_fecha_hora(r.kickoff, zona, getattr(r, "hora_confirmada", True)), tema=tema)

    p_vis, p_loc = par_enteros(r.p_away_pre)
    fav_vis = float(r.p_away_pre) > float(r.p_home_pre)
    a, h = r.away_team, r.home_team
    for team, cx, rol, fav in ((a, 270, "visitante", fav_vis), (h, 810, "local", not fav_vis)):
        _logo(fig, team, cx, 620, 76)
        _texto(fig, cx, 706, team, 32, TEXTO, bold=fav, ha="center")
        _texto(fig, cx, 744, rol, 21, TEXTO2, ha="center")
    _texto(fig, 540, 620, "@", 32, TENUE, ha="center")

    _barra(fig, MARGEN, 806, ANCHO_PX - MARGEN, 898,
           [(p_vis, AZUL if fav_vis else LINEA), (p_loc, LINEA if fav_vis else AZUL)], 40)
    p_fav, fav = (p_vis, a) if fav_vis else (p_loc, h)
    _texto(fig, 540, 932, f"Aun con {p_fav}%, {fav} no gana {100 - p_fav} de cada 100 veces",
           15, TENUE, ha="center")

    # Tabla comparativa: en negritas el valor que favorece
    filas = []
    if a in rec and h in rec:
        filas.append(("RÉCORD", fmt_record(rec[a]), fmt_record(rec[h]),
                      np.sign(_pct_victorias(rec[a]) - _pct_victorias(rec[h]))))
    ea, eh = float(r.elo_away_pre), float(r.elo_home_pre)
    filas.append(("ELO", _miles(ea), _miles(eh), np.sign(round(ea) - round(eh))))
    if a in rank and h in rank:
        filas.append(("RANKING ELO", f"#{rank[a]}", f"#{rank[h]}", np.sign(rank[h] - rank[a])))
    y = 990
    _hline(fig, MARGEN, ANCHO_PX - MARGEN, y)
    for nombre, va_, vh, mejor in filas:
        yc = y + 28
        _texto(fig, 270, yc, va_, 30, TEXTO if mejor > 0 else TEXTO2, bold=mejor > 0, ha="center")
        _texto(fig, 540, yc, nombre, 16, TENUE, bold=True, ha="center")
        _texto(fig, 810, yc, vh, 30, TEXTO if mejor < 0 else TEXTO2, bold=mejor < 0, ha="center")
        y += 56
        _hline(fig, MARGEN, ANCHO_PX - MARGEN, y)

    # Bajas y lesiones (solo Modelo 2: games trae bajas_* y ajuste_elo_*)
    ys = y + 96
    if hasattr(r, "bajas_home"):
        ys = _bloque_bajas(fig, r, y + 32) + 76
    _seccion(fig, ys, "!", "Lo que ve el modelo", "¿A qué hay que ponerle atención?")
    _texto(fig, MARGEN, ys + 66, _envolver(explicacion, 60), 25, TEXTO, va="top",
           linespacing=1.5)
    return _guardar(fig, ruta)


# ══════════════════════════════════════════════════════════════════════════════
# 7 · Resumen en texto (para el guion)
# ══════════════════════════════════════════════════════════════════════════════
def _pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


def escribir_resumen(ruta, semana, threshold, tabla, calib, ev, week_games, destacados, zona,
                     seccion_modelos=None, titulo=None, nota_calibracion=None):
    L = [titulo or f"# Semana {semana} · Pronosticando la NFL con un modelo de ELO", ""]
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
        L += ["", "### Calibración (temporada)", ""]
        if nota_calibracion:
            L += [f"_Historial publicado · {nota_calibracion[1]} "
                  "(gráfica 11_calibracion)._", ""]
        L += ["| Confianza | Partidos | Récord | Prob. media | % real |",
              "|---|---|---|---|---|"]
        for c in calib.itertuples():
            if c.partidos:
                L.append(f"| {c.rango} | {c.partidos} | {c.record} | "
                         f"{c.prob_media:.0%} | {c.pct_real:.0%} |")
        L.append("")

    if seccion_modelos:
        L += seccion_modelos + [""]

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


def comparar_modelos(games, m2, semana, season, threshold, carpeta, registrar=True,
                     origen="semanal"):
    """
    Modelo 1 (Elo) vs Modelo 2 (Elo + ausencias):
      1. Congela en el libro los pronósticos de ambos para `semana` (si registrar).
      2. Evalúa lo congelado de las semanas anteriores (comparacion_modelos.csv).
      3. Tabla de la semana con los ausentes y el ajuste (ausencias_semana_XX.csv).
      4. Regresa las líneas de la sección "Modelo 1 vs Modelo 2" para resumen.md.
    """
    g2 = m2["enriquecido"]
    params = f"{m2['modo']}|{m2['params'].get('calibrado_en') or 'sin calibrar'}"
    if registrar:
        n1 = libro_pron.registrar_pronosticos(games, "elo_base", semana, season, origen,
                                              params="base")
        n2 = libro_pron.registrar_pronosticos(g2, "elo_ausencias", semana, season, origen,
                                              params=params)
        ruta = libro_pron.ruta_libro(season)
        print(f"Libro: +{len(n1)} pronósticos Modelo 1 · +{len(n2)} Modelo 2 "
              f"→ {ruta.relative_to(ROOT) if ruta.is_relative_to(ROOT) else ruta}")
    else:
        print("Libro: sin cambios (--sin-actualizar)")

    libro = libro_pron.cargar_libro(season)
    tabla, ev = libro_pron.evaluar_modelos(libro, games, threshold, hasta_semana=semana)
    res = libro_pron.resumen_comparacion(tabla)
    des = libro_pron.desacuerdos(ev)
    if len(tabla):
        tabla.to_csv(carpeta / "comparacion_modelos.csv", index=False)
    if len(des):
        des.to_csv(carpeta / "desacuerdos_modelos.csv", index=False)

    # ── La semana de los picks: dónde difieren y por qué ─────────────────────
    sem = g2[g2["week"] == semana].merge(
        games[["game_id", "p_home_pre", "favorito", "p_fav"]].rename(
            columns={"p_home_pre": "p_home_m1", "favorito": "favorito_m1", "p_fav": "p_fav_m1"}),
        on="game_id")
    sem = sem.rename(columns={"p_home_pre": "p_home_m2", "favorito": "favorito_m2",
                              "p_fav": "p_fav_m2"})
    sem["dif_p_home"] = (sem["p_home_m2"] - sem["p_home_m1"]).round(4)
    cols = ["week", "kickoff", "away_team", "home_team", "p_home_m1", "p_home_m2", "dif_p_home",
            "favorito_m1", "p_fav_m1", "favorito_m2", "p_fav_m2", "ajuste_elo_home",
            "ajuste_elo_away", "ausentes_home", "ausentes_away", "fuente_reporte"]
    sem = sem.sort_values(["kickoff", "game_id"], na_position="last")[cols]
    sem.to_csv(carpeta / f"ausencias_semana_{semana:02d}.csv", index=False)

    # ── Texto ────────────────────────────────────────────────────────────────
    L = ["## Modelo 1 (Elo) vs Modelo 2 (Elo + ausencias)", ""]
    if len(res):
        w = int(res["hasta_semana"].iat[0])
        vig = libro_pron.pronostico_vigente(libro)
        backfill = vig[(vig["week"] < semana) & (vig["origen"] == "backfill")]["week"]
        L += [f"Pronósticos congelados antes de cada kickoff · semanas 1–{w} · "
              f"{int(res['partidos'].iat[0])} partidos · pick si prob. > {threshold:.0%}", "",
              "| Modelo | Récord | % aciertos | Brier | Log loss |", "|---|---|---|---|---|"]
        for r in res.itertuples():
            L.append(f"| {r.nombre} | {r.record} | {_pct(r.pct_aciertos)} | "
                     f"{r.brier:.3f} | {r.log_loss:.3f} |")
        if pd.notna(res["vegas_brier"].iat[0]):
            L.append(f"| Las Vegas (referencia) | — | {_pct(res['vegas_pct'].iat[0])} | "
                     f"{res['vegas_brier'].iat[0]:.3f} | — |")
        mejor = res.sort_values("brier").iloc[0]
        L += ["", f"- Mejor Brier hasta ahora: **{mejor.nombre}** (más bajo = mejor)."]
        if len(des):
            c = des["acerto"].value_counts()
            L.append(f"- Eligieron distinto favorito en {len(des)} partidos → acertó el "
                     f"Modelo 1 en {c.get('Modelo 1', 0)} y el Modelo 2 en {c.get('Modelo 2', 0)}.")
        if len(backfill):
            L.append(f"- Semanas {int(backfill.min())}–{int(backfill.max())}: reconstruidas "
                     "(backfill) con el reporte final de lesiones de cada semana.")
    else:
        L += ["Aún no hay partidos jugados con pronóstico congelado de ambos modelos "
              "(corre `python3 src/compare_models.py --backfill` para las semanas ya jugadas)."]

    fuente = sem["fuente_reporte"].mode().iat[0] if len(sem) else "—"
    dif = sem[(sem["dif_p_home"].abs() >= 0.02) | (sem["favorito_m1"] != sem["favorito_m2"])]
    dif = dif.reindex(dif["dif_p_home"].abs().sort_values(ascending=False).index)
    L += ["", f"### Semana {semana}: dónde difieren (ausencias según {fuente})", ""]
    if not len(dif):
        L.append("- Los dos modelos dan prácticamente lo mismo en todos los partidos.")
    for r in dif.itertuples():
        cambio = " · **cambia el favorito**" if r.favorito_m1 != r.favorito_m2 else ""
        L.append(f"- {r.away_team} @ {r.home_team}: Modelo 1 {r.favorito_m1} {r.p_fav_m1:.0%} · "
                 f"Modelo 2 {r.favorito_m2} {r.p_fav_m2:.0%}{cambio}")
        for equipo, aus, aj in ((r.away_team, r.ausentes_away, r.ajuste_elo_away),
                                (r.home_team, r.ausentes_home, r.ajuste_elo_home)):
            if isinstance(aus, str) and aus:
                L.append(f"    - {equipo} ({aj:+.0f} Elo): {aus}")
    if str(fuente).startswith("provisional"):
        L += ["", "_Ojo: el reporte de lesiones de esta semana aún no sale. Vuelve a correr "
              "`python3 src/weekly_content.py --solo-pronosticos` el viernes/sábado para "
              "congelar la versión final del Modelo 2._"]

    for linea in L:
        if linea.startswith("|") or linea.startswith("- Mejor") or linea.startswith("- Eligieron"):
            print("   ", linea)
    return {"tabla": tabla, "resumen": res, "evaluacion": ev, "desacuerdos": des,
            "semana": sem, "seccion": L}


def escribir_comentarios(ruta, semana, partidos, rec, rank, zona):
    """comentarios.md — texto listo para copiar en cada historia."""
    L = [f"# Semana {semana} · Historias por partido", ""]
    for i, r, texto in partidos:
        L += [f"## {i:02d} · {r.away_team} @ {r.home_team} — "
              f"{_fmt_fecha_hora(r.kickoff, zona, getattr(r, 'hora_confirmada', True))}", "",
              f"Favorito: **{r.favorito} {r.p_fav:.0%}** · Elo {r.elo_away_pre:,.0f} "
              f"(#{rank.get(r.away_team, '–')}) vs {r.elo_home_pre:,.0f} "
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
                      historias=True, formatos=None, ausencias=True, modo=None,
                      solo_pronosticos=False, imagenes_modelo_2=True):
    """
    Corre todo el flujo semanal y regresa un dict con los DataFrames y las rutas.

    semana    : semana de los picks. None = la próxima por jugarse. Las métricas
                usan solo las semanas anteriores (lo que se sabía en ese momento).
    historias : True = una imagen por partido en partidos/ + comentarios.md.
    formatos  : tamaños a generar (claves de FORMATOS). None = todos.
    ausencias : True = corre también el Modelo 2 (Elo + ausencias), congela los
                pronósticos de ambos en el libro (data/processed/pronosticos/) y
                agrega la comparación al resumen. Las imágenes usan el Modelo 1.
    modo      : 'overlay' | 'integrado' para el Modelo 2 (None = el calibrado).
    solo_pronosticos : re-registra la semana con el reporte de lesiones más
                reciente y escribe la comparación, sin regenerar imágenes
                (pensado para correrlo viernes/sábado).
    imagenes_modelo_2 : True = además genera el set completo de imágenes, métricas
                y resumen.md del Modelo 2 en reels/semana_XX/modelo_2/ (el pie
                de cada imagen dice "ELO + AUSENCIAS").
    """
    global LOGOS_DIR
    if logos_dir:
        LOGOS_DIR = Path(logos_dir)
    if not 0.5 <= threshold < 1:
        raise ValueError("threshold debe estar entre 0.50 y 1 (ej. 0.5, 0.6)")
    if solo_pronosticos and not ausencias:
        raise ValueError("--solo-pronosticos necesita el Modelo 2 (quita --sin-ausencias)")

    print("── 1 · Datos " + "─" * 60)
    schedule, final_elos = cargar_datos(season)
    games_df, elo_evolution_df = calcular_elo(schedule, final_elos)
    if actualizar:
        guardar_archivos(games_df, elo_evolution_df, season)
    games = enriquecer(games_df, schedule, zona)

    m2 = None
    if ausencias:
        from src.elo_ausencias import modelo_ausencias_temporada
        print("── 1b · Modelo 2: Elo + ausencias " + "─" * 39)
        m2 = modelo_ausencias_temporada(schedule, final_elos, season, modo)
        if actualizar:
            m2["games"].to_csv(DATA_DIR / f"weekly_predictions_ausencias_{season}.csv", index=False)
            m2["elo_evolution"].to_csv(DATA_DIR / f"elo_evolution_ausencias_{season}.csv",
                                       index=False)
            print(f"Guardado: weekly_predictions_ausencias_{season}.csv, "
                  f"elo_evolution_ausencias_{season}.csv")
        m2["enriquecido"] = enriquecer(m2["games"], schedule, zona)

    if semana is None:
        semana = int(current_week_probabilities(games_df)["week"].iloc[0])
    elo_tab = tabla_elo(elo_evolution_df)
    if semana not in elo_tab.columns:
        raise ValueError(f"La semana {semana} no está en el calendario de {season}")
    carpeta = Path(out_dir) if out_dir else REELS_DIR / f"semana_{semana:02d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"Semana de los picks : {semana}  →  {carpeta}")

    seccion_modelos = None
    if m2 is not None:
        print("── 1c · Libro de pronósticos (Modelo 1 vs Modelo 2) " + "─" * 21)
        comp = comparar_modelos(games, m2, semana, season, threshold, carpeta,
                                registrar=actualizar,
                                origen="actualizacion" if solo_pronosticos else "semanal")
        seccion_modelos = comp["seccion"]
        if solo_pronosticos:
            ruta = carpeta / "modelos.md"
            ruta.write_text("\n".join(comp["seccion"]) + "\n", encoding="utf-8")
            print("✓", ruta.relative_to(ROOT) if ruta.is_relative_to(ROOT) else ruta)
            return {"semana": semana, "games": games, "modelo_2": m2, "comparacion": comp,
                    "carpeta": carpeta}

    print("── 2-3 · Modelo 1 (Elo) · métricas e imágenes " + "─" * 27)
    out = metricas_e_imagenes(games, elo_tab, semana, carpeta, threshold, zona, top_n,
                              historias, formatos, seccion_modelos)

    if m2 is not None and imagenes_modelo_2:
        print("── 2-3 · Modelo 2 (Elo + ausencias) · métricas e imágenes " + "─" * 15)
        # Calibración del historial PUBLICADO: Modelo 1 antes del cambio, Modelo 2 desde él
        calib_pub, nota = None, None
        cambio = CAMBIO_MODELO.get(season)
        if cambio is not None and semana >= cambio:
            ev_pub = libro_pron.evaluar_publicado(libro_pron.cargar_libro(season), games,
                                                  cambio, threshold, hasta_semana=semana)
            if len(ev_pub):
                calib_pub = calibracion(ev_pub)
                nota = nota_cambio_modelo(semana, cambio)
                print(f"   Calibración publicada: Modelo 1 sem. 1–{cambio - 1} + "
                      f"Modelo 2 sem. {cambio}–{semana - 1} ({len(ev_pub)} partidos)"
                      if semana > cambio else
                      f"   Calibración publicada: Modelo 1 sem. 1–{cambio - 1} "
                      f"({len(ev_pub)} partidos) + aviso del cambio")
        with usar_modelo("Elo + ausencias"):
            out2 = metricas_e_imagenes(
                m2["enriquecido"], tabla_elo(m2["elo_evolution"]), semana, carpeta / "modelo_2",
                threshold, zona, top_n, historias, formatos, seccion_modelos,
                titulo=f"# Semana {semana} · Modelo 2: Elo + ausencias",
                calib_override=calib_pub, nota_calibracion=nota)
        m2["salida"] = out2
        out["rutas"] += out2["rutas"]

    return {"semana": semana, "games": games, "carpeta": carpeta, "modelo_2": m2, **out}


def metricas_e_imagenes(games, elo_tab, semana, carpeta, threshold, zona, top_n=TOP_N,
                        historias=True, formatos=None, seccion_modelos=None, titulo=None,
                        calib_override=None, nota_calibracion=None):
    """
    Métricas (semanas < semana), todas las imágenes y resumen.md de UN modelo.
    games: enriquecer() del modelo · elo_tab: tabla_elo() de su evolución de Elo.
    calib_override: calibración a usar en 11_calibracion.png, calibracion.csv y
        resumen.md en lugar de la del propio modelo (la del modelo se guarda
        aparte en calibracion_modelo.csv). nota_calibracion: texto de la gráfica.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    print("   Métricas")
    ev = evaluar_partidos(games[games["week"] < semana], threshold)
    tabla = metricas_por_semana(ev) if len(ev) else pd.DataFrame()
    calib = calibracion(ev)
    if calib_override is not None:
        calib.to_csv(carpeta / "calibracion_modelo.csv", index=False)
        calib = calib_override
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

    print("   Gráficas")
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
                    "Quién sube, quién cae", "Cambio de Elo", f"En la semana {semana - 1}",
                    img / f"02_elo_semana_{semana - 1:02d}.png", n=top_n))
            if semana > 2:   # en la semana 2 sería idéntica a la anterior
                rutas.append(graf_suben_caen(
                    elo_tab[semana] - elo_tab[elo_tab.columns.min()], records(games, semana),
                    "Quién sube, quién cae", "Cambio de Elo", "Desde la pretemporada",
                    img / "03_elo_temporada.png", n=top_n))
            rutas.append(graf_top(elo_tab[semana], img / "04_top_elo.png", n=RANKING_N,
                                  elo_previo=elo_tab[semana - 1] if semana > 1 else None,
                                  semana=semana))
            rutas.append(graf_picks(week_games, semana, threshold,
                                    img / f"{semana - 1:02d}_picks_semana_{semana:02d}.png", zona))
            for i, (etiqueta, r, texto) in enumerate(destacados, start=6):
                nombre = f"{i:02d}_{_slug(etiqueta)}_{r.away_team}_{r.home_team}.png"
                rutas.append(graf_duelo(r, etiqueta, texto, img / nombre, zona, rec, rank,
                                        tema=f"Pronósticos · semana {semana}"))
            if semana > 1:
                rutas.append(graf_evolucion_elo(elo_tab, semana, img / "09_evolucion_elo.png",
                                                n=top_n))
                rutas.append(graf_evolucion_divisiones(elo_tab, semana,
                                                       img / "10_evolucion_divisiones.png"))
            if calib["partidos"].sum():
                rutas.append(graf_calibracion(calib, img / "11_calibracion.png",
                                              nota=nota_calibracion))

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
                        zona, rec, rank, tema=f"Pronósticos · semana {semana}"))
                rutas.append(escribir_comentarios(img / "partidos" / "comentarios.md",
                                                  semana, partidos, rec, rank, zona))
    rutas.append(escribir_resumen(carpeta / "resumen.md", semana, threshold, tabla, calib,
                                  ev, week_games, destacados, zona, seccion_modelos, titulo,
                                  nota_calibracion))
    for r in rutas:
        print("✓", r.relative_to(ROOT) if r.is_relative_to(ROOT) else r)

    return {"evaluacion": ev, "metricas": tabla, "calibracion": calib,
            "destacados": destacados, "rutas": rutas}


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
    p.add_argument("--sin-ausencias", action="store_true",
                   help="Solo el Modelo 1 (Elo base), como antes: sin Modelo 2 ni libro")
    p.add_argument("--solo-pronosticos", action="store_true",
                   help="Re-congela los pronósticos de la semana con el reporte de lesiones "
                        "más reciente y escribe la comparación, sin regenerar imágenes")
    p.add_argument("--modo", choices=["overlay", "integrado"], default=None,
                   help="Modo del Modelo 2 (default: el que eligió la calibración)")
    p.add_argument("--sin-imagenes-modelo-2", action="store_true",
                   help="No genera las imágenes del Modelo 2 en semana_XX/modelo_2/")
    a = p.parse_args(argv)
    plt.switch_backend("Agg")
    generar_contenido(season=a.season, threshold=a.threshold, semana=a.semana,
                      actualizar=not a.sin_actualizar, out_dir=a.out, zona=a.zona,
                      logos_dir=a.logos_dir, top_n=a.top, historias=not a.sin_historias,
                      formatos=a.formatos, ausencias=not a.sin_ausencias, modo=a.modo,
                      solo_pronosticos=a.solo_pronosticos,
                      imagenes_modelo_2=not a.sin_imagenes_modelo_2)


if __name__ == "__main__":
    main()
