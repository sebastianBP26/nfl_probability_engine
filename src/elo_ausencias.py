# src/elo_ausencias.py
"""
Modelo 2 · Elo + ausencias
==========================

Misma idea del Elo calibrado (mismos HOME_ADVANTAGE, K, regresión y media),
con un ajuste antes de cada partido por los titulares que faltan:

    diff = Elo_local − Elo_visita + HA  −  Σ_g δ_g · A_local,g  +  Σ_g δ_g · A_visita,g
    P(local) = 1 / (1 + 10^(−diff / 400))

    g      ∈ {QB, RB, WRTE, OL, DEF}
    A_t,g  titulares del grupo g que se espera que falten (ver src/availability.py):
           suma de P(no juega) de cada titular según el reporte de lesiones + IR.
    δ_g    puntos Elo que cuesta UN titular ausente del grupo g. Se estiman con
           datos (calibrar_deltas) y se guardan en data/processed/params_ausencias.json.

Dos formas de actualizar el Elo después del partido (parámetro `modo`):
    overlay    el Elo evoluciona igual que el modelo base; el ajuste solo cambia
               la probabilidad publicada. (Es lo que hace el parche del QB.)
    integrado  la actualización usa la probabilidad ajustada (estilo FiveThirtyEight):
               si un equipo pierde sin su QB, se le castiga menos porque ya se
               esperaba que perdiera más seguido.

El modelo sigue siendo SOLO probabilidad de victoria (nunca contra el spread).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.elo import apply_season_regression, expected_win_prob, update_elo
from src.availability import GRUPOS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
PARAMS_PATH = DATA_DIR / "params_ausencias.json"

# Mismos parámetros Elo que src/weekly_content.py (Notebook 02)
HOME_ADVANTAGE = 42.2
K_REGULAR = 40
K_PLAYOFFS = 48
REGRESSION_FACTOR = 0.40
GLOBAL_MEAN = 1503.1
GAME_TYPE_K = {"REG": K_REGULAR, "WC": K_PLAYOFFS, "DIV": K_PLAYOFFS,
               "CON": K_PLAYOFFS, "SB": K_PLAYOFFS}

# Valores por defecto si aún no existe params_ausencias.json (QB = parche auditado)
DELTAS_DEFAULT = {"QB": 90.0, "RB": 0.0, "WRTE": 0.0, "OL": 0.0, "DEF": 0.0}
MODO_DEFAULT = "overlay"


# ══════════════════════════════════════════════════════════════════════════════
# Parámetros
# ══════════════════════════════════════════════════════════════════════════════
def cargar_parametros(path=PARAMS_PATH):
    """dict con deltas, modo, prob_ausencia y metadatos de la calibración."""
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"deltas": dict(DELTAS_DEFAULT), "modo": MODO_DEFAULT, "prob_ausencia": None,
            "calibrado_en": None, "nota": "valores por defecto (sin calibrar)"}


def guardar_parametros(params, path=PARAMS_PATH):
    params["calibrado_en"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"Guardado: {Path(path).relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Ajuste
# ══════════════════════════════════════════════════════════════════════════════
def ajuste_equipo(A, deltas):
    """Puntos Elo que pierde un equipo: −Σ δ_g · A_g. A: dict {grupo: ausencias}."""
    return -float(sum(deltas.get(g, 0.0) * float(A.get(g, 0.0) or 0.0) for g in GRUPOS))


def ajustar_probabilidad(p_home, ajuste_home, ajuste_away):
    """
    Ajusta una probabilidad ya calculada (como el parche del QB): reconstruye la
    diferencia de Elo desde p y le suma el ajuste neto. Si los dos equipos
    pierden lo mismo, la probabilidad no cambia.
    """
    neto = ajuste_home - ajuste_away
    if neto == 0:
        return p_home
    p = min(max(p_home, 1e-6), 1 - 1e-6)
    diff = -400.0 * np.log10(1.0 / p - 1.0) + neto
    return float(1.0 / (1.0 + 10.0 ** (-diff / 400.0)))


def _matriz_ausencias(games, panel, col="pre"):
    """
    Para cada partido de `games` (game_id, home_team, away_team) regresa dos
    matrices n×5 con las ausencias del local y del visitante (columnas
    A_<col>_<grupo> del panel; 0 si no hay datos).
    """
    cols = [f"A_{col}_{g}" for g in GRUPOS]
    p = panel.set_index(["game_id", "team"])[cols]
    idx_h = pd.MultiIndex.from_arrays([games["game_id"], games["home_team"]])
    idx_a = pd.MultiIndex.from_arrays([games["game_id"], games["away_team"]])
    Ah = p.reindex(idx_h).fillna(0.0).to_numpy(dtype=float)
    Aa = p.reindex(idx_a).fillna(0.0).to_numpy(dtype=float)
    return Ah, Aa


def _deltas_vec(deltas):
    return np.array([float(deltas.get(g, 0.0)) for g in GRUPOS])


# ══════════════════════════════════════════════════════════════════════════════
# Histórico (backtest / calibración)
# ══════════════════════════════════════════════════════════════════════════════
def preparar_historico(df, panel, col="pre"):
    """
    Arreglos numpy del histórico (schedules limpio de src/elo.py:load_nfl_data,
    ordenado como run_elo_system) + matrices de ausencias. Se calcula una vez
    y se reutiliza en cada evaluación de la calibración.
    """
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    Ah, Aa = _matriz_ausencias(df, panel, col)
    y = np.where(df["home_score"] == df["away_score"], 0.5,
                 (df["home_score"] > df["away_score"]).astype(float))
    return {
        "df": df,
        "teams": teams,
        "h": df["home_team"].map(t_idx).to_numpy(),
        "a": df["away_team"].map(t_idx).to_numpy(),
        "season": df["season"].to_numpy(),
        "result": df["home_team_win"].to_numpy(dtype=float),   # empate = 0, como el base
        "y": y,                                                 # empate = 0.5 para métricas
        "k": df["game_type"].map(GAME_TYPE_K).to_numpy(dtype=float),
        "ha": np.where(df["game_type"] == "SB", 0.0, HOME_ADVANTAGE),
        "X": Aa - Ah,   # ausencias netas: positivo = le faltan más al visitante
        "Ah": Ah,
        "Aa": Aa,
    }


def correr_historico(H, deltas, modo="overlay"):
    """
    Corre el Elo partido a partido sobre el histórico preparado.
    Regresa (p_base, p_ajustada, elos_finales_dict). En modo overlay las
    trayectorias de Elo son idénticas al modelo base.
    """
    d = _deltas_vec(deltas)
    neto = H["X"] @ d                     # puntos Elo a favor del local
    n_teams = len(H["teams"])
    elo = np.full(n_teams, 1500.0)
    p_base = np.empty(len(neto))
    p_adj = np.empty(len(neto))
    season_prev = None
    for i in range(len(neto)):
        s = H["season"][i]
        if s != season_prev:
            if season_prev is not None:
                elo = (1 - REGRESSION_FACTOR) * elo + REGRESSION_FACTOR * GLOBAL_MEAN
                elo = np.round(elo, 4)
            season_prev = s
        h, a = H["h"][i], H["a"][i]
        diff = elo[h] - elo[a] + H["ha"][i]
        pb = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        pa = 1.0 / (1.0 + 10.0 ** (-(diff + neto[i]) / 400.0))
        p_base[i], p_adj[i] = pb, pa
        p_upd = pa if modo == "integrado" else pb
        delta = H["k"][i] * (H["result"][i] - p_upd)
        elo[h] = round(elo[h] + delta, 4)
        elo[a] = round(elo[a] - delta, 4)
    return p_base, p_adj, dict(zip(H["teams"], elo))


def metricas(y, p, mask=None):
    """Brier, log loss y % de aciertos del favorito (empate = medio acierto)."""
    if mask is not None:
        y, p = y[mask], p[mask]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    acierto = np.where(y == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float))
    return {
        "partidos": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "aciertos": float(np.mean(acierto)),
    }


def _log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibrar_deltas(H, mask, grupos=GRUPOS, modo="overlay", x0=None, fijos=None,
                    cota=(0.0, 300.0)):
    """
    δ que minimizan el log loss en los partidos `mask` (entrenamiento).
    Restricción δ ≥ 0 (que falte un titular nunca ayuda). `fijos` deja grupos
    en un valor fijo (p. ej. {"RB": 0}).
    """
    from scipy.optimize import minimize

    fijos = fijos or {}
    libres = [g for g in grupos if g not in fijos]
    base = {g: 0.0 for g in GRUPOS}
    base.update(fijos)

    if modo == "overlay":
        p_base, _, _ = correr_historico(H, {g: 0.0 for g in GRUPOS}, "overlay")
        d0 = -400.0 * np.log10(1.0 / np.clip(p_base, 1e-9, 1 - 1e-9) - 1.0)
        cols = [GRUPOS.index(g) for g in libres]
        Xl = H["X"][mask][:, cols]
        fijo_neto = H["X"][mask] @ _deltas_vec(base)
        y = H["y"][mask]
        d0m = d0[mask]
        c = np.log(10) / 400.0

        def f(v):
            z = d0m + fijo_neto + Xl @ v
            p = np.clip(1.0 / (1.0 + 10.0 ** (-z / 400.0)), 1e-9, 1 - 1e-9)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            grad = -(c * (y - p)) @ Xl / len(y)
            return loss, grad

        jac = True
    else:
        def f(v):
            dd = dict(base)
            dd.update(zip(libres, v))
            _, p, _ = correr_historico(H, dd, "integrado")
            return _log_loss(H["y"][mask], p[mask])

        jac = False

    x0 = np.array([50.0] * len(libres) if x0 is None else x0, dtype=float)
    # En modo integrado el Elo se redondea a 4 decimales en cada partido: con el
    # paso por defecto (1e-8) el gradiente numérico sale 0. Paso de 0.5 puntos.
    opciones = {} if jac else {"eps": 0.5}
    res = minimize(f, x0, jac=jac, method="L-BFGS-B", bounds=[cota] * len(libres),
                   options=opciones)
    out = dict(base)
    out.update({g: round(float(v), 1) for g, v in zip(libres, res.x)})
    return out


def bootstrap_deltas(H, mask, n=200, semilla=26, **kw):
    """Intervalos (percentil 5-95) remuestreando partidos de entrenamiento (modo overlay)."""
    rng = np.random.default_rng(semilla)
    idx = np.flatnonzero(mask)
    filas = []
    for _ in range(n):
        rep = rng.choice(idx, size=len(idx), replace=True)
        filas.append(_calibrar_overlay_indices(H, rep, **kw))
    t = pd.DataFrame(filas)
    return pd.DataFrame({"p05": t.quantile(0.05), "mediana": t.median(), "p95": t.quantile(0.95)})


def _calibrar_overlay_indices(H, rep, grupos=GRUPOS, cota=(0.0, 300.0)):
    """Calibración overlay sobre índices con repetición (para el bootstrap)."""
    from scipy.optimize import minimize

    if "_d0" not in H:
        p_base, _, _ = correr_historico(H, {g: 0.0 for g in GRUPOS}, "overlay")
        H["_d0"] = -400.0 * np.log10(1.0 / np.clip(p_base, 1e-9, 1 - 1e-9) - 1.0)
    cols = [GRUPOS.index(g) for g in grupos]
    X = H["X"][rep][:, cols]
    y = H["y"][rep]
    d0 = H["_d0"][rep]
    c = np.log(10) / 400.0

    def f(v):
        z = d0 + X @ v
        p = np.clip(1.0 / (1.0 + 10.0 ** (-z / 400.0)), 1e-9, 1 - 1e-9)
        return (-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                -(c * (y - p)) @ X / len(y))

    res = minimize(f, np.full(len(cols), 50.0), jac=True, method="L-BFGS-B",
                   bounds=[cota] * len(cols))
    return dict(zip(grupos, res.x))


# ══════════════════════════════════════════════════════════════════════════════
# Temporada en curso — espejo de src/weekly_probabilities.py:build_weekly_snapshots
# ══════════════════════════════════════════════════════════════════════════════
def elos_iniciales(final_elos_base, modo, season):
    """
    Elo de cierre de la temporada anterior para el modelo 2. En modo overlay es
    el mismo que el base; en modo integrado se usa el archivo que deja la
    calibración (elo_final_ausencias_<season-1>.csv), porque su historia difiere.
    """
    if modo != "integrado":
        return final_elos_base
    path = DATA_DIR / f"elo_final_ausencias_{season - 1}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} no existe: corre "
                                "`python3 src/compare_models.py --recalibrar`.")
    prev = pd.read_csv(path)
    return dict(zip(prev["team"], prev["elo_final"]))


def build_weekly_snapshots_ausencias(schedule_df, final_elos_prev_season, panel, deltas,
                                     modo="overlay", home_advantage=HOME_ADVANTAGE,
                                     k_regular=K_REGULAR, k_playoffs=K_PLAYOFFS,
                                     regression_factor=REGRESSION_FACTOR,
                                     global_mean=GLOBAL_MEAN):
    """
    Igual que build_weekly_snapshots (mismas columnas y mismo orden), con el
    ajuste por ausencias. p_home_pre / p_away_pre ya vienen ajustadas; además:
        p_home_base      probabilidad sin ajuste (con el Elo de este modelo)
        ajuste_elo_home  puntos Elo que pierde el local por ausencias (≤ 0)
        ajuste_elo_away  ídem visitante
        ausentes_home / ausentes_away / fuente_reporte  (para revisar a mano)
        bajas_home / bajas_away   todas las bajas, aunque su grupo tenga δ = 0
    """
    game_type_k = {"REG": k_regular, "WC": k_playoffs, "DIV": k_playoffs,
                   "CON": k_playoffs, "SB": k_playoffs}
    sort_cols = ["week", "gameday"] if "gameday" in schedule_df.columns else ["week"]
    sched = (schedule_df.dropna(subset=["week"]).sort_values(sort_cols)
             .reset_index(drop=True).copy())

    pan = panel.set_index(["game_id", "team"])
    cols_A = [f"A_pre_{g}" for g in GRUPOS]
    # Solo se listan ausentes de grupos que sí mueven la probabilidad (δ > 0)
    activos = tuple(g for g in GRUPOS if deltas.get(g, 0.0) > 0)

    def _filtrar(texto):
        if not isinstance(texto, str) or not texto:
            return ""
        return "; ".join(t for t in texto.split("; ")
                         if any(f"({g}," in t for g in activos))

    def _info(game_id, team):
        if (game_id, team) in pan.index:
            r = pan.loc[(game_id, team)]
            A = {g: float(r[f"A_pre_{g}"]) for g in GRUPOS}
            todos = r["ausentes"] if isinstance(r["ausentes"], str) else ""
            return A, _filtrar(todos), r["fuente_reporte"], todos
        return {g: 0.0 for g in GRUPOS}, "", "sin datos", ""

    elos = apply_season_regression(final_elos_prev_season, regression_factor, global_mean)
    game_records, elo_records = [], []

    for week in sorted(sched["week"].unique()):
        week_games = sched[sched["week"] == week]
        for team, elo in elos.items():
            elo_records.append({"week": int(week), "team": team, "elo_pre_week": round(elo, 2)})

        for g in week_games.itertuples(index=False):
            home, away = g.home_team, g.away_team
            ha = 0.0 if g.game_type == "SB" else home_advantage
            k = game_type_k.get(g.game_type, k_regular)
            A_h, aus_h, fuente, bajas_h = _info(g.game_id, home)
            A_a, aus_a, _, bajas_a = _info(g.game_id, away)
            aj_h, aj_a = ajuste_equipo(A_h, deltas), ajuste_equipo(A_a, deltas)

            elo_home_pre, elo_away_pre = elos[home], elos[away]
            p_base = expected_win_prob(elo_home_pre, elo_away_pre, ha)
            p_home = expected_win_prob(elo_home_pre + aj_h, elo_away_pre + aj_a, ha)

            played = pd.notna(g.home_score) and pd.notna(g.away_score)
            result = None
            elo_home_post, elo_away_post = elo_home_pre, elo_away_pre
            if played:
                result = int(g.home_score > g.away_score)
                if modo == "integrado":
                    delta = k * (result - p_home)
                    elo_home_post = round(elo_home_pre + delta, 4)
                    elo_away_post = round(elo_away_pre - delta, 4)
                else:
                    elo_home_post, elo_away_post = update_elo(
                        elo_home_pre, elo_away_pre, result, k, ha)
                elos[home], elos[away] = elo_home_post, elo_away_post

            game_records.append({
                "season": g.season, "week": int(week), "game_type": g.game_type,
                "home_team": home, "away_team": away,
                "elo_home_pre": round(elo_home_pre, 2), "elo_away_pre": round(elo_away_pre, 2),
                "p_home_pre": round(p_home, 4), "p_away_pre": round(1 - p_home, 4),
                "played": played, "home_score": g.home_score, "away_score": g.away_score,
                "result": result,
                "elo_home_post": round(elo_home_post, 2), "elo_away_post": round(elo_away_post, 2),
                "p_home_base": round(p_base, 4),
                "ajuste_elo_home": round(aj_h, 1), "ajuste_elo_away": round(aj_a, 1),
                **{f"home_{c}": round(A_h[c[6:]], 2) for c in cols_A},
                **{f"away_{c}": round(A_a[c[6:]], 2) for c in cols_A},
                "ausentes_home": aus_h, "ausentes_away": aus_a, "fuente_reporte": fuente,
                # Todas las bajas (también las de grupos que el modelo no pondera)
                "bajas_home": bajas_h, "bajas_away": bajas_a,
            })

    return pd.DataFrame(game_records), pd.DataFrame(elo_records)


def modelo_ausencias_temporada(schedule, final_elos_base, season, modo=None, params=None,
                               cambios=None):
    """
    Todo lo que necesita el flujo semanal para el modelo 2:
        games, elo_evolution  (mismo formato que build_weekly_snapshots)
        panel, info           (ausencias por equipo-partido y último reporte)
        params, modo
    """
    from src.availability import ausencias_temporada

    params = params or cargar_parametros()
    modo = modo or params.get("modo", MODO_DEFAULT)
    panel, info = ausencias_temporada(season, probs=params.get("prob_ausencia"), cambios=cambios)
    elos0 = elos_iniciales(final_elos_base, modo, season)
    games, evo = build_weekly_snapshots_ausencias(schedule, elos0, panel, params["deltas"], modo)
    d = params["deltas"]
    print(f"Modelo 2 · modo {modo} · δ Elo por titular ausente: "
          + ", ".join(f"{g} {d.get(g, 0):.0f}" for g in GRUPOS)
          + f"  (calibrado: {params.get('calibrado_en') or 'no'})")
    print(f"Último reporte de lesiones: semana {info['ultima_semana_reporte']} · "
          f"rosters: semana {info['ultima_semana_roster']} · snaps: semana {info['ultima_semana_snaps']}")
    sug = info.get("sugerencias_qb")
    if sug is not None and len(sug):
        print("⚠ QB titular sin iniciar en los últimos 3 partidos (si es permanente, "
              "regístralo en CAMBIOS_TITULAR de src/availability.py):")
        for r in sug.itertuples():
            print(f"    {r.team}: titular {r.qb_titular} · inicia {r.qb_actual}")
    return {"games": games, "elo_evolution": evo, "panel": panel, "info": info,
            "params": params, "modo": modo}


# ══════════════════════════════════════════════════════════════════════════════
# Autoprueba (casos del parche original del QB, δ_QB = 90)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    d = {"QB": 90.0}
    aj = ajuste_equipo({"QB": 1.0}, d)
    assert aj == -90.0
    p = ajustar_probabilidad(0.70, aj, 0.0)
    assert abs(p - 0.5816) < 0.002, p
    assert abs(ajustar_probabilidad(0.50, aj, 0.0) - 0.3733) < 0.002
    assert abs(ajustar_probabilidad(0.7731, aj, aj) - 0.7731) < 1e-9
    assert ajustar_probabilidad(0.80, 0.0, 0.0) == 0.80
    # El cálculo desde Elo da lo mismo que reconstruir desde p
    pe = expected_win_prob(1650 + aj, 1500, HOME_ADVANTAGE)
    pr = ajustar_probabilidad(expected_win_prob(1650, 1500, HOME_ADVANTAGE), aj, 0.0)
    assert abs(pe - pr) < 1e-9
    print("OK: autoprueba del parche (δ_QB = 90) superada.")
