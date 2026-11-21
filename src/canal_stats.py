# src/canal_stats.py
"""
Láminas semanales para el canal de IG · acumulado de la temporada
=================================================================

Enfoque 1 (nivel general): una lámina por posición (QB, RB, WR, TE, DEF, K)
y un resumen tipo dashboard. Enfoque 2 (nivel partido): una lámina por
partido de la semana con la probabilidad del Modelo 2 (Elo + ausencias),
ataque y defensa lado a lado, jugadores clave y el duelo clave.

Todo con el acumulado de las semanas 1 … N-1 y el branding Black Sea de
src/weekly_content.py (mismo lienzo, paleta, pie y formatos size_1 / size_2).

Fuentes:
    nflreadpy  load_player_stats · load_team_stats · load_schedules
    data/processed/pronosticos/pronosticos_<season>.csv
        Pronóstico CONGELADO del Modelo 2 (model_registry.pronostico_vigente,
        modelo == "elo_ausencias"); nunca se recalcula ni se sustituye por el
        Modelo 1.

Uso:
    python3 src/canal_stats.py                    # la próxima semana por jugarse
    python3 src/canal_stats.py --semana 10 --formatos size_2 --sin-partidos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model_registry as libro_pron  # noqa: E402
from src import player_stats as ps  # noqa: E402
from src.weekly_content import (  # noqa: E402
    ANCHO_PX, AZUL, AZUL_TXT, BLANCO, ESTILO, FORMATOS, LINEA, MARGEN, NEUTRO, PANEL,
    ROJO, ROJO_TXT, SEASON, TENUE, TEXTO, TEXTO2, Y_LINEA, ZONA_HORARIA, ALTA_CONFIANZA,
    _ancho_texto, _barra, _caja, _e, _fmt_fecha_hora, _guardar, _hline, _lienzo, _logo,
    _miles, _pildora, _seccion, _texto, par_decimal, usar_formato, usar_modelo,
)

CANAL_DIR = ROOT / "canal"
DER = ANCHO_PX - MARGEN
MIN_PCT_PARTIDOS = 0.60   # para entrar a una tabla: jugó >= 60% de los partidos de su equipo
TOP = 5
MODELO_2 = "elo_ausencias"

# Qué se muestra de cada posición
#   lider       métrica con la que se ordena el acumulado (total)
#   extra       segunda columna (total)
#   constante   métrica de la tabla de constancia
#   menor       True si en esa métrica menos es mejor (defensa)
POSICIONES = {
    "QB": dict(titulo="Quarterbacks", lider="passing_yards", extra="passing_tds",
               constante="passing_yards", col_lider="YARDAS", col_extra="TD", unidad="yardas"),
    "RB": dict(titulo="Running backs", lider="scrimmage_yards", extra="carries",
               constante="carries", col_lider="YDS TOT", col_extra="ACARREOS",
               unidad="acarreos"),
    "WR": dict(titulo="Receptores abiertos", lider="receiving_yards", extra="receptions",
               constante="receptions", col_lider="YARDAS", col_extra="REC",
               unidad="recepciones"),
    "TE": dict(titulo="Alas cerradas", lider="receiving_yards", extra="receptions",
               constante="receptions", col_lider="YARDAS", col_extra="REC",
               unidad="recepciones"),
    "DEF": dict(titulo="Defensivas", lider="pts_permitidos", extra="capturas",
                constante="pts_permitidos", col_lider="PTS PERM", col_extra="CAPTURAS",
                unidad="puntos permitidos", menor=True),
    "K": dict(titulo="Pateadores", lider="fg_made", extra="pat_made",
              constante="puntos_k", col_lider="FG", col_extra="PAT", unidad="puntos"),
}
UNIDAD_CORTA = {"passing_yards": "yds", "scrimmage_yards": "yds", "receiving_yards": "yds",
                "carries": "acarreos", "receptions": "rec", "pts_permitidos": "pts",
                "puntos_k": "pts", "fg_made": "FG"}


# ══════════════════════════════════════════════════════════════════════════════
# 1 · Datos
# ══════════════════════════════════════════════════════════════════════════════
def proxima_semana(cal):
    """Primera semana de temporada regular con partidos sin marcador."""
    reg = cal[cal["game_type"] == "REG"]
    pend = reg[reg["home_score"].isna()]
    return int(pend["week"].min()) if len(pend) else int(reg["week"].max()) + 1


def cargar(season, semana):
    """Todo lo necesario para la semana: jugadores, equipos, calendario y Modelo 2."""
    cal = ps.cargar_calendario(season)
    jug = ps.cargar_player_stats(season, semanas=(1, semana - 1))
    pe = ps.partidos_equipo(season, semana - 1, cal)
    libro = libro_pron.pronostico_vigente(libro_pron.cargar_libro(season))
    m2 = libro[(libro["modelo"] == MODELO_2) & (libro["week"] == semana)] if len(libro) else libro
    return dict(cal=cal, jug=jug, pe=pe, m2=m2.set_index("game_id") if len(m2) else m2)


def _min_partidos(pe):
    """Partidos mínimos para entrar a una tabla (60% de los juegos de su equipo)."""
    juegos = pe.groupby("team").size()
    return juegos, max(2, int(np.ceil(MIN_PCT_PARTIDOS * juegos.median())))


def tabla_posicion(datos, pos):
    """
    Resumen por jugador (o por equipo en DEF) con las métricas de la lámina.
    Devuelve (lideres, constantes, completo).
    """
    cfg = POSICIONES[pos]
    juegos, min_pj = _min_partidos(datos["pe"])
    if pos == "DEF":
        d = datos["pe"].assign(nombre=datos["pe"]["team"])
        base = d.rename(columns={"team": "id"}).assign(team=d["team"])
        idc, nomc = "id", "nombre"
    else:
        base = ps.datos_posicion(datos["jug"], pos)
        idc, nomc = "player_id", "player_display_name"
    r = ps.resumen_jugadores(base, cfg["lider"], 1, idc, nomc)
    ext = base.groupby(idc)[cfg["extra"]].sum()
    r["extra"] = r[idc].map(ext).values
    cons = ps.resumen_jugadores(base, cfg["constante"], 1, idc, nomc).set_index(idc)
    for c in ["media", "std", "cv", "p20", "p80", "minimo", "maximo"]:
        r[f"c_{c}"] = r[idc].map(cons[c]).values
    if pos == "K":
        att = base.groupby(idc)["fg_att"].sum()
        r["fg_att"] = r[idc].map(att).values
        r["fg_long"] = r[idc].map(base.groupby(idc)["fg_long"].max()).values
    # mínimo de partidos relativo a los juegos de SU equipo
    r["juegos_equipo"] = r["equipo"].map(juegos).fillna(juegos.median())
    r = r[r["partidos"] >= np.maximum(min_pj, np.ceil(MIN_PCT_PARTIDOS * r["juegos_equipo"]))]

    menor = cfg.get("menor", False)
    if menor:   # defensa: menos puntos permitidos por partido es mejor
        lideres = r.sort_values(["media", "extra"], ascending=[True, False])
    elif pos == "K":
        lideres = r.sort_values(["total", "extra"], ascending=False)
    else:
        lideres = r.sort_values(["total", "media"], ascending=False)
    # Constantes: la mitad buena de la posición, ordenada por menor CV
    med = r["c_media"].median()
    buenos = r[r["c_media"] <= med] if menor else r[r["c_media"] >= med]
    constantes = buenos.sort_values("c_cv")
    return lideres.head(TOP), constantes.head(TOP), r, base


def dato_destacado(constantes, base, pos):
    """
    Para el más constante: cuántos de sus partidos alcanzaron su "piso"
    (p20 redondeado). Devuelve (cifra, leyenda, nota) o None.
    """
    if constantes.empty:
        return None
    cfg = POSICIONES[pos]
    f = constantes.iloc[0]
    idc = "id" if pos == "DEF" else "player_id"
    serie = base.loc[base[idc] == f[idc], cfg["constante"]]
    if cfg.get("menor"):
        t = int(np.ceil(f["c_p80"]))
        hits = int((serie <= t).sum())
        leyenda = f"PARTIDOS DE {f['jugador']} PERMITIENDO {t} PUNTOS O MENOS"
    else:
        t = int(np.floor(f["c_p20"]))
        hits = int((serie >= t).sum())
        leyenda = f"PARTIDOS DE {_corto(f['jugador']).upper()} CON {t} O MÁS {cfg['unidad'].upper()}"
    n = int(serie.size)
    fallos = n - hits
    nota = (f"Aun así, {fallos} de {n} partidos quedó fuera: una racha no garantiza el siguiente"
            if fallos else f"Son solo {n} partidos: una muestra chica no garantiza el siguiente")
    return f"{hits}/{n}", leyenda, nota


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Utilidades de texto
# ══════════════════════════════════════════════════════════════════════════════
def _corto(nombre):
    """'Kenneth Walker III' -> 'K. Walker III' (equipos se quedan igual)."""
    partes = str(nombre).split()
    if len(partes) < 2 or len(nombre) <= 3:
        return str(nombre)
    return f"{partes[0][0]}. {' '.join(partes[1:])}"


def _ajustar(s, px, ancho, bold=True, minimo=14):
    """Tamaño de fuente (px) para que `s` quepa en `ancho` px."""
    while px > minimo and _ancho_texto(s, px, bold) > ancho:
        px -= 1
    return px


def _num(v, dec=1):
    return _miles(v) if abs(v) >= 1000 or dec == 0 else f"{v:.{dec}f}"


def _ancho_pildora():
    """Píldora a todo lo ancho (952 px reales) en cualquier formato."""
    return (ANCHO_PX - 2 * MARGEN) / _e(1)


CORTO_DUELO = {"passing_yards": "YARDAS POR PASE", "scrimmage_yards": "YARDAS TOTALES",
               "receiving_yards": "YARDAS RECIBIDAS"}


def _nombre_fila(fig, x, y, nombre, px, ancho, bold):
    _texto(fig, x, y, nombre, _ajustar(nombre, px, ancho, bold), TEXTO, bold=bold)


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Enfoque 1 · láminas por posición
# ══════════════════════════════════════════════════════════════════════════════
FILA = 50


def _tabla_lideres(fig, y, lideres, pos):
    cfg = POSICIONES[pos]
    cols = [(560, "PJ"), (720, cfg["col_lider"]), (870, "POR PJ"), (DER, cfg["col_extra"])]
    if pos == "K":
        cols = [(560, "PJ"), (720, "FG"), (870, "% FG"), (DER, "LARGO")]
    if pos == "DEF":
        cols = [(560, "PJ"), (720, "PTS/PJ"), (870, "YDS/PJ"), (DER, "CAPTURAS")]
    _texto(fig, 100, y, "#", 16, TENUE, bold=True, ha="center")
    _texto(fig, 152, y, "EQUIPO" if pos == "DEF" else "JUGADOR", 16, TENUE, bold=True)
    for x, t in cols:
        _texto(fig, x, y, t, 16, TENUE, bold=True, ha="right")
    y += 22
    _hline(fig, MARGEN, DER, y)
    for i, f in enumerate(lideres.itertuples()):
        lider = i == 0
        yc = y + FILA / 2
        if lider:
            _caja(fig, MARGEN, y + 1, DER, y + FILA, PANEL)
        _texto(fig, 100, yc, f"{i + 1}", 24, ROJO_TXT if lider else TEXTO2, bold=True, ha="center")
        _logo(fig, f.equipo, 160, yc, 32)
        nombre = f.jugador if pos == "DEF" else _corto(f.jugador)
        _nombre_fila(fig, 190, yc, nombre, 24, 300, lider)
        if pos == "K":
            vals = [f"{f.partidos}", f"{f.total:.0f}/{f.fg_att:.0f}",
                    f"{100 * f.total / max(f.fg_att, 1):.0f}%", f"{f.fg_long:.0f}"]
        elif pos == "DEF":   # YDS/PJ se escribe en graf_posicion
            vals = [f"{f.partidos}", f"{f.media:.1f}", None, f"{f.extra:.0f}"]
        else:
            vals = [f"{f.partidos}", _miles(f.total), f"{f.media:.1f}", f"{f.extra:.0f}"]
        for j, ((x, _), v) in enumerate(zip(cols, vals)):
            if v is None:
                continue
            hero = lider and j == 1
            _texto(fig, x, yc, v, 24, ROJO_TXT if hero else TEXTO, bold=hero or lider, ha="right")
        y += FILA
        _hline(fig, MARGEN, DER, y)
    return y


def _tabla_constantes(fig, y, constantes, pos):
    cfg = POSICIONES[pos]
    u = UNIDAD_CORTA.get(cfg["constante"], "")
    bx0, bx1 = 600, 860
    _texto(fig, 100, y, "#", 16, TENUE, bold=True, ha="center")
    _texto(fig, 152, y, "EQUIPO" if pos == "DEF" else "JUGADOR", 16, TENUE, bold=True)
    _texto(fig, 560, y, "MEDIA ± DE", 16, TENUE, bold=True, ha="right")
    _texto(fig, (bx0 + bx1) / 2, y, "RANGO HABITUAL", 16, TENUE, bold=True, ha="center")
    _texto(fig, DER, y, "CV", 16, TENUE, bold=True, ha="right")
    y += 22
    _hline(fig, MARGEN, DER, y)
    if constantes.empty:
        return y
    lo = min(0.0, float(constantes["c_minimo"].min()))
    hi = float(constantes["c_maximo"].max()) or 1.0
    esc = lambda v: bx0 + (bx1 - bx0) * (v - lo) / (hi - lo)
    for i, f in enumerate(constantes.itertuples()):
        lider = i == 0
        yc = y + FILA / 2
        if lider:
            _caja(fig, MARGEN, y + 1, DER, y + FILA, PANEL)
        _texto(fig, 100, yc, f"{i + 1}", 24, TEXTO if lider else TEXTO2, bold=True, ha="center")
        _logo(fig, f.equipo, 160, yc, 32)
        nombre = f.jugador if pos == "DEF" else _corto(f.jugador)
        _nombre_fila(fig, 190, yc, nombre, 24, 230, lider)
        _texto(fig, 560, yc, f"{f.c_media:.1f} ± {f.c_std:.1f}", 22, TEXTO, bold=lider, ha="right")
        # pista = min a max de la tabla · relleno = p20 a p80 · marca = media
        _caja(fig, bx0, yc - 4, bx1, yc + 4, LINEA)
        _caja(fig, esc(f.c_p20), yc - 10, max(esc(f.c_p80), esc(f.c_p20) + 4), yc + 10, AZUL)
        _caja(fig, esc(f.c_media) - 2, yc - 14, esc(f.c_media) + 2, yc + 14, TEXTO)
        _texto(fig, DER, yc, f"{f.c_cv:.2f}", 22, TEXTO if lider else TEXTO2, bold=lider, ha="right")
        y += FILA
        _hline(fig, MARGEN, DER, y)
    _texto(fig, MARGEN, y + 24, f"Barra: 60% central de sus partidos ({u}) · línea blanca: "
           "media · CV = DE / media (menos = más constante)", 15, TENUE)
    return y + 24


def graf_posicion(datos, pos, semana, ruta):
    cfg = POSICIONES[pos]
    lideres, constantes, _, base = tabla_posicion(datos, pos)
    sem = f"Semanas 1–{semana - 1}" if semana > 2 else "Semana 1"
    fig = _lienzo(f"Acumulado · {pos}", cfg["titulo"], f"{sem} · temporada regular",
                  tema=f"Estadísticas · semana {semana}")
    pregunta = {
        "DEF": "¿Quién permite menos puntos por partido?",
        "K": "¿Quién ha metido más goles de campo?",
    }.get(pos, f"¿Quién suma más {ps.nombre(cfg['lider']).lower()}?")
    _seccion(fig, 560, "#", "Líderes del acumulado", pregunta)
    y = _tabla_lideres(fig, 625, lideres, pos)
    if pos == "DEF":   # yardas permitidas por partido en la tabla de líderes
        yds = datos["pe"].groupby("team")["yds_permitidas"].mean()
        yy = 625 + 22
        for f in lideres.itertuples():
            _texto(fig, 870, yy + FILA / 2, f"{yds.get(f.equipo, np.nan):.0f}", 24,
                   TEXTO, bold=False, ha="right")
            yy += FILA

    y0 = y + 70
    _seccion(fig, y0, "%", "Los más constantes",
             f"¿Quién varía menos en {ps.nombre(cfg['constante']).lower()} por partido?")
    y = _tabla_constantes(fig, y0 + 65, constantes, pos)

    dato = dato_destacado(constantes, base, pos)
    if dato:
        cifra, leyenda, nota = dato
        _pildora(fig, 540, y + 40, cifra, leyenda, ancho=_ancho_pildora(), alto=130)
        _texto(fig, 540, y + 200, nota, 15, TENUE, ha="center")
    return _guardar(fig, ruta), lideres, constantes


def graf_resumen(datos, semana, ruta):
    """Dashboard 2 x 3: el líder y el más constante de cada posición."""
    sem = f"semanas 1–{semana - 1}" if semana > 2 else "semana 1"
    fig = _lienzo("El acumulado de la temporada", "Seis posiciones, un vistazo",
                  f"Líderes y los más constantes · {sem}",
                  tema=f"Estadísticas · semana {semana}")
    ancho = (DER - MARGEN - 24) / 2
    alto, gap, top = 330, 22, 540
    for k, pos in enumerate(POSICIONES):
        cfg = POSICIONES[pos]
        lideres, constantes, _, _ = tabla_posicion(datos, pos)
        x0 = MARGEN + (k % 2) * (ancho + 24)
        y0 = top + (k // 2) * (alto + gap)
        x1 = x0 + ancho
        _caja(fig, x0, y0, x1, y0 + alto, PANEL, radio=12)
        _texto(fig, x0 + 24, y0 + 34, f"{pos} · {ps.nombre(cfg['lider']).upper()}"
               if pos != "DEF" else "DEF · MENOS PUNTOS PERMITIDOS", 15, TENUE, bold=True)
        if lideres.empty:
            continue
        f = lideres.iloc[0]
        _logo(fig, f["equipo"], x0 + 70, y0 + 118, 60)
        nombre = f["jugador"] if pos == "DEF" else _corto(f["jugador"])
        _texto(fig, x0 + 128, y0 + 94, nombre, _ajustar(nombre, 26, ancho - 150), TEXTO, bold=True)
        if pos == "DEF":
            cifra, sub = f"{f['media']:.1f}", "puntos permitidos por partido"
        elif pos == "K":
            cifra, sub = f"{f['total']:.0f}/{f['fg_att']:.0f}", f"goles de campo · largo {f['fg_long']:.0f}"
        else:
            cifra = _miles(f["total"])
            sub = f"{UNIDAD_CORTA[cfg['lider']]} · {f['media']:.1f} por partido"
        _texto(fig, x0 + 128, y0 + 142, cifra, 44, TEXTO, bold=True)
        _texto(fig, x0 + 24, y0 + 196, sub, 18, TEXTO2)
        _hline(fig, x0 + 24, x1 - 24, y0 + 228)
        _texto(fig, x0 + 24, y0 + 258, "MÁS CONSTANTE", 14, TENUE, bold=True)
        if len(constantes):
            c = constantes.iloc[0]
            u = UNIDAD_CORTA.get(cfg["constante"], "")
            txt = (f"{c['jugador'] if pos == 'DEF' else _corto(c['jugador'])} · "
                   f"{c['c_media']:.1f} ± {c['c_std']:.1f} {u}")
            _texto(fig, x0 + 24, y0 + 294, txt, _ajustar(txt, 20, ancho - 48, False), AZUL_TXT)
    _texto(fig, 540, top + 3 * alto + 2 * gap + 28,
           "± = desviación estándar por partido · mínimo 60% de los partidos de su equipo",
           15, TENUE, ha="center")
    return _guardar(fig, ruta)


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Enfoque 2 · lámina por partido
# ══════════════════════════════════════════════════════════════════════════════
CLAVES = [("QB", "attempts", "passing_yards", "passing_tds", "yds"),
          ("RB", "touches", "scrimmage_yards", "total_tds", "yds"),
          ("WR", "targets", "receiving_yards", "total_tds", "yds"),
          ("TE", "targets", "receiving_yards", "total_tds", "yds")]


def jugadores_clave(jug, equipo):
    """QB, RB1, WR1 y TE1 del equipo (más volumen con ese equipo) y su perfil."""
    filas = []
    for pos, vol, met, td, u in CLAVES:
        d = ps.datos_posicion(jug, pos)
        d = d[d["team"] == equipo]
        if d.empty:
            continue
        pid = d.groupby("player_id")[vol].sum().idxmax()
        s = d[d["player_id"] == pid]
        filas.append(dict(pos=pos, player_id=pid, jugador=s["player_display_name"].iloc[-1],
                          metrica=met, media=s[met].mean(), std=s[met].std(ddof=1),
                          partidos=len(s), td_rate=(s[td] >= 1).mean(), unidad=u))
    return pd.DataFrame(filas)


def duelo_clave(jug, a, b, claves):
    """
    El jugador clave con el mejor cruce: qué tanto produce respecto a la mediana
    de su posición x qué tanto permite la defensa rival a esa posición.
    Solo jugadores en la mitad alta de su posición y defensas que permiten más
    que el promedio.
    """
    mejor = None
    for equipo, rival in ((a, b), (b, a)):
        for f in claves[equipo].itertuples():
            fac = ps.factor_defensa(jug, f.pos, f.metrica).get(rival, np.nan)
            med = ps.resumen_jugadores(ps.datos_posicion(jug, f.pos), f.metrica, 3)["media"].median()
            if pd.isna(fac) or fac <= 1 or f.media < med:
                continue
            score = fac * f.media / med
            if mejor is None or score > mejor[-1]:
                mejor = (fac, f, equipo, rival, score)
    return mejor[:4] if mejor else None


COMPARA = [("pts", "Puntos por partido", False), ("yds", "Yardas por partido", False),
           ("pts_permitidos", "Puntos permitidos", True),
           ("yds_permitidas", "Yardas permitidas", True),
           ("capturas", "Capturas por partido", False),
           ("entregas", "Entregas forzadas", False)]


def graf_partido(datos, r, i, n, semana, ruta, zona=ZONA_HORARIA):
    a, h = r["away_team"], r["home_team"]
    kick = r.get("kickoff")
    fecha = _fmt_fecha_hora(kick.tz_convert(zona) if pd.notna(kick) else kick, zona)
    fig = _lienzo(f"Semana {semana} · Partido {i} de {n}", f"{a} @ {h}", fecha,
                  tema=f"Estadísticas · semana {semana}")

    # ── Encabezado: escudos + probabilidad del Modelo 2 ──────────────────────
    _logo(fig, a, 200, 590, 76)
    _logo(fig, h, 880, 590, 76)
    _texto(fig, 540, 590, "@", 30, TENUE, bold=True, ha="center")
    m2 = datos["m2"]
    gid = r["game_id"]
    if len(m2) and gid in m2.index:
        p_a, p_h = par_decimal(m2.loc[gid, "p_away"])
        fav_a = p_a > p_h
        p_fav = max(p_a, p_h)
        c_fav = ROJO if p_fav >= ALTA_CONFIANZA * 100 else AZUL
        _barra(fig, MARGEN, 660, DER, 716,
               [(p_a, c_fav if fav_a else LINEA), (p_h, LINEA if fav_a else c_fav)], 26, 1)
        fav, otro = (a, h) if fav_a else (h, a)
        nota = (f"Aun con {p_fav:.1f}%, {fav} no gana {round(100 - p_fav)} de cada 100 veces")
    else:
        _barra(fig, MARGEN, 660, DER, 716, [(100, NEUTRO)], 26, 0)
        _texto(fig, 540, 688, "Sin pronóstico", 24, BLANCO, bold=True, ha="center")
        nota = "El Modelo 2 aún no congela un pronóstico para este partido"
    _texto(fig, MARGEN, 742, "PROBABILIDAD DE VICTORIA · MODELO 2 (ELO + AUSENCIAS)", 15,
           TENUE, bold=True)
    vegas = ""
    if pd.notna(r.get("spread_line")):
        sp = float(r["spread_line"])
        fav_v = h if sp > 0 else a
        vegas = f"Las Vegas: {fav_v} -{abs(sp):g}" if sp else "Las Vegas: pick'em"
        if pd.notna(r.get("total_line")):
            vegas += f" · total {float(r['total_line']):g}"
    _texto(fig, DER, 742, vegas, 15, TENUE, ha="right")
    _texto(fig, 540, 778, nota, 15, TENUE, ha="center")

    # ── Ataque y defensa lado a lado ─────────────────────────────────────────
    _seccion(fig, 850, "#", "Ataque y defensa", "¿Quién llega mejor en lo que va de la temporada?")
    eq = ps.resumen_equipos(datos["pe"])
    y = 910
    _logo(fig, a, 110, y, 32)
    _logo(fig, h, 970, y, 32)
    _texto(fig, 540, y, "PROMEDIO POR PARTIDO", 16, TENUE, bold=True, ha="center")
    y += 24
    _hline(fig, MARGEN, DER, y)
    FIL = 44
    for col, etiqueta, menor in COMPARA:
        va, vh = eq.at[a, col] if a in eq.index else np.nan, eq.at[h, col] if h in eq.index else np.nan
        yc = y + FIL / 2
        tope = np.nanmax([va, vh, 1e-9])
        mejor_a = (va < vh) if menor else (va > vh)
        empate = va == vh
        dec = 0 if col in ("yds", "yds_permitidas") else 1
        for v, x_val, x_ancla, signo, es_mejor in ((va, 110, 380, -1, mejor_a and not empate),
                                                   (vh, 970, 700, 1, not mejor_a and not empate)):
            if pd.isna(v):
                continue
            largo = 150 * v / tope
            x0, x1 = sorted((x_ancla, x_ancla + signo * largo))
            _caja(fig, x0, yc - 9, x1, yc + 9, AZUL if es_mejor else LINEA)
            _texto(fig, x_val, yc, _num(v, dec), 22, TEXTO if es_mejor else TEXTO2,
                   bold=es_mejor, ha="center")
        _texto(fig, 540, yc, etiqueta, 18, TEXTO2, ha="center")
        y += FIL
        _hline(fig, MARGEN, DER, y)

    # ── Jugadores clave ──────────────────────────────────────────────────────
    _seccion(fig, y + 50, "!", "Jugadores clave",
             "¿Cuánto produce cada uno por partido? (media ± DE · % de partidos con TD)")
    claves = {t: jugadores_clave(datos["jug"], t) for t in (a, h)}
    y += 100
    ancho = (DER - MARGEN - 40) / 2
    for k, t in enumerate((a, h)):
        x0 = MARGEN + k * (ancho + 40)
        yy = y
        for f in claves[t].itertuples():
            yc = yy + 21
            _texto(fig, x0, yc, f.pos, 15, TENUE, bold=True)
            _texto(fig, x0 + 44, yc, _corto(f.jugador), _ajustar(_corto(f.jugador), 19, 170), TEXTO,
                   bold=True)
            std = 0 if pd.isna(f.std) else f.std
            txt = f"{f.media:.0f} ± {std:.0f} {f.unidad} · {100 * f.td_rate:.0f}%"
            _texto(fig, x0 + ancho, yc, txt, _ajustar(txt, 18, ancho - 220, False), TEXTO2,
                   ha="right")
            yy += 42
            _hline(fig, x0, x0 + ancho, yy)
    y += 4 * 42

    # ── Duelo clave ──────────────────────────────────────────────────────────
    duelo = duelo_clave(datos["jug"], a, h, claves)
    if duelo:
        fac, f, equipo, rival = duelo
        dif = 100 * (fac - 1)
        cifra = f"{dif:+.0f}%"
        quien = {"QB": "AL QB", "RB": "A LOS RB", "WR": "A LOS WR", "TE": "A LOS TE"}[f.pos]
        leyenda = (f"{rival} PERMITE {quien} {dif:.0f}% MÁS {CORTO_DUELO[f.metrica]} QUE EL "
                   f"PROMEDIO · {_corto(f.jugador).upper()} PROMEDIA {f.media:.0f}")
        _pildora(fig, 540, y + 22, cifra, leyenda, ancho=_ancho_pildora(), alto=120)
    else:
        _texto(fig, 540, y + 70, "Ningún jugador clave enfrenta una defensa más permisiva "
               "que el promedio", 18, TENUE, ha="center")
    return _guardar(fig, ruta), claves, duelo


# ══════════════════════════════════════════════════════════════════════════════
# 5 · Flujo semanal
# ══════════════════════════════════════════════════════════════════════════════
ARCHIVOS = {"QB": "01_qb", "RB": "02_rb", "WR": "03_wr", "TE": "04_te", "DEF": "05_def", "K": "06_k"}


def partidos_semana(datos, semana):
    cal = datos["cal"]
    g = cal[(cal["week"] == semana) & (cal["game_type"] == "REG")].copy()
    m2 = datos["m2"]
    kick = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns, UTC]")
    if len(m2):
        kick = pd.to_datetime(g["game_id"].map(m2["kickoff"]), utc=True, errors="coerce")
    g["kickoff"] = kick
    return g.sort_values(["kickoff", "game_id"], na_position="last").reset_index(drop=True)


def _resumen_md(ruta, semana, tablas, partidos):
    lineas = [f"# Canal · semana {semana}", "",
              f"Acumulado de las semanas 1–{semana - 1} (temporada regular). "
              "Fuente: nflverse (nflreadpy). Pronósticos: Modelo 2 (Elo + ausencias), "
              "congelados en el libro de pronósticos. No es consejo de apuesta.", ""]
    for pos, (lid, con) in tablas.items():
        lineas.append(f"## {pos} · {POSICIONES[pos]['titulo']}")
        if len(lid):
            f = lid.iloc[0]
            lineas.append(f"- Líder: **{f['jugador']}** ({f['equipo']}) · total {f['total']:.0f} · "
                          f"{f['media']:.1f} por partido en {f['partidos']} partidos")
        if len(con):
            c = con.iloc[0]
            lineas.append(f"- Más constante: **{c['jugador']}** ({c['equipo']}) · "
                          f"{c['c_media']:.1f} ± {c['c_std']:.1f} · CV {c['c_cv']:.2f}")
        lineas.append("")
    if partidos:
        lineas += ["## Partidos", ""]
        for txt in partidos:
            lineas.append(f"- {txt}")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def generar_canal(season=SEASON, semana=None, formatos=None, partidos=True, out_dir=None):
    plt.switch_backend("Agg")
    cal = ps.cargar_calendario(season)
    semana = semana or proxima_semana(cal)
    print(f"Canal · temporada {season} · semana {semana} (acumulado 1–{semana - 1})")
    datos = cargar(season, semana)
    carpeta = Path(out_dir) if out_dir else CANAL_DIR / f"semana_{semana:02d}"
    (carpeta / "tablas").mkdir(parents=True, exist_ok=True)
    juegos = partidos_semana(datos, semana) if partidos else pd.DataFrame()

    tablas, txt_partidos, rutas = {}, [], []
    for nombre_fmt in formatos or FORMATOS:
        img = carpeta / nombre_fmt
        with plt.rc_context(ESTILO), usar_formato(FORMATOS[nombre_fmt]):
            for pos, archivo in ARCHIVOS.items():
                ruta, lid, con = graf_posicion(datos, pos, semana, img / f"{archivo}.png")
                rutas.append(ruta)
                tablas[pos] = (lid, con)
            rutas.append(graf_resumen(datos, semana, img / "07_resumen.png"))
            txt_partidos = []
            with usar_modelo("Modelo 2 · Elo + ausencias"):
                for i, r in juegos.iterrows():
                    ruta, claves, duelo = graf_partido(
                        datos, r, i + 1, len(juegos), semana,
                        img / "partidos" / f"{i + 1:02d}_{r.away_team}_{r.home_team}.png")
                    rutas.append(ruta)
                    t = f"{r.away_team} @ {r.home_team}"
                    if duelo:
                        fac, f, _, rival = duelo
                        t += (f" · duelo clave: {f.jugador} ({f.pos}) vs defensa de {rival}, "
                              f"que permite {100 * (fac - 1):+.0f}% en {ps.nombre(f.metrica).lower()}")
                    txt_partidos.append(t)

    # Fuentes de cada cifra
    for pos in POSICIONES:
        _, _, completo, _ = tabla_posicion(datos, pos)
        completo.to_csv(carpeta / "tablas" / f"acumulado_{pos.lower()}.csv", index=False)
    ps.resumen_equipos(datos["pe"]).to_csv(carpeta / "tablas" / "equipos.csv")
    if len(juegos):
        cols = ["game_id", "away_team", "home_team", "kickoff", "spread_line", "total_line"]
        j = juegos[cols].copy()
        if len(datos["m2"]):
            j["p_away_modelo2"] = j["game_id"].map(datos["m2"]["p_away"])
            j["p_home_modelo2"] = j["game_id"].map(datos["m2"]["p_home"])
        j.to_csv(carpeta / "tablas" / "partidos.csv", index=False)
    rutas.append(_resumen_md(carpeta / "resumen.md", semana, tablas, txt_partidos))
    for r in rutas:
        print("✓", r.relative_to(ROOT) if r.is_relative_to(ROOT) else r)
    return rutas


def main(argv=None):
    p = argparse.ArgumentParser(description="Láminas del canal de IG · acumulado por posición")
    p.add_argument("--season", type=int, default=SEASON)
    p.add_argument("--semana", type=int, default=None,
                   help="Semana por jugarse; el acumulado llega hasta la anterior. "
                        "Default: la próxima semana")
    p.add_argument("--formatos", nargs="+", choices=list(FORMATOS), default=None)
    p.add_argument("--sin-partidos", action="store_true", help="Solo las láminas por posición")
    p.add_argument("--out", default=None, help="Carpeta de salida (default: canal/semana_XX)")
    a = p.parse_args(argv)
    generar_canal(a.season, a.semana, a.formatos, not a.sin_partidos, a.out)


if __name__ == "__main__":
    main()
