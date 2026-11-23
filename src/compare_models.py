#!/usr/bin/env python3
# src/compare_models.py
"""
Modelo 1 (Elo) vs Modelo 2 (Elo + ausencias) · calibración, backfill y tabla
============================================================================

    python3 src/compare_models.py --recalibrar   # re-estima δ y P(ausencia) con 2012-2025
    python3 src/compare_models.py --backfill     # congela en el libro las semanas ya jugadas
    python3 src/compare_models.py --tabla        # comparación acumulada desde el libro

--recalibrar (tarda ~1 min; descarga 2012-hoy de nflreadpy)
    1. Panel de ausencias por equipo-partido 2012-hoy (src/availability.py).
    2. P(no juega | estatus del reporte), con titulares 2012-2025.
    3. δ por grupo minimizando log loss en 2012-2021 con las ausencias REALES
       (efecto de que falte un titular). Qué grupos entran se decide con
       validación cruzada dejando una temporada fuera, solo dentro de 2012-2021.
    4. Prueba fuera de muestra 2022-2025 con la información PREVIA al partido
       (reporte del viernes y, aparte, del martes), contra el Elo base, el
       parche QB (δ=90) y Las Vegas. Bootstrap pareado de la mejora en Brier.
    5. Modo (overlay / integrado) = el de menor Brier fuera de muestra.
    6. δ finales re-estimados con 2012-2025 -> data/processed/params_ausencias.json
       + backtest_ausencias.csv, prob_ausencia.csv, ausencias_panel.csv y
       elo_final_ausencias_2025.csv (Elo de arranque del modo integrado).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.availability import (  # noqa: E402
    GRUPOS, agregar_panel, cargar_fuentes, estimar_prob_ausencia, tabla_a_probs, tabla_larga,
)
from src import elo_ausencias as ea  # noqa: E402
from src import model_registry as libro_pron  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
SEASON = 2026
TRAIN = (2012, 2021)
TEST = (2022, 2025)


# ══════════════════════════════════════════════════════════════════════════════
# Recalibración
# ══════════════════════════════════════════════════════════════════════════════
def _mask(H, rango):
    return (H["season"] >= rango[0]) & (H["season"] <= rango[1])


def _p_vegas(df):
    def imp(ml):
        ml = pd.to_numeric(ml, errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    ph, pa = imp(df["home_moneyline"]), imp(df["away_moneyline"])
    return ph / (ph + pa)


def _cv_grupos(H, rango):
    """
    Validación cruzada dejando una temporada fuera dentro de `rango`.
    Regresa el log loss medio con todos los grupos y quitando cada uno.
    """
    temporadas = range(rango[0], rango[1] + 1)
    configs = {"todos": {}}
    configs.update({f"sin_{g}": {g: 0.0} for g in GRUPOS if g != "QB"})
    filas = []
    for nombre, fijos in configs.items():
        ll = []
        for s in temporadas:
            tr = _mask(H, rango) & (H["season"] != s)
            te = H["season"] == s
            d = ea.calibrar_deltas(H, tr, fijos=fijos)
            _, p, _ = ea.correr_historico(H, d)
            ll.append(ea.metricas(H["y"], p, te)["log_loss"] * te.sum())
        n = sum(int((H["season"] == s).sum()) for s in temporadas)
        filas.append({"config": nombre, "log_loss_cv": sum(ll) / n})
    return pd.DataFrame(filas)


def _bootstrap_mejora(y, p_base, p_mod, n=2000, semilla=26):
    """Bootstrap pareado (por partido) de Brier(base) − Brier(modelo)."""
    rng = np.random.default_rng(semilla)
    e_b, e_m = (p_base - y) ** 2, (p_mod - y) ** 2
    idx = rng.integers(0, len(y), size=(n, len(y)))
    dif = e_b[idx].mean(axis=1) - e_m[idx].mean(axis=1)
    return {"mejora_brier": float(e_b.mean() - e_m.mean()),
            "ic90_bajo": float(np.quantile(dif, 0.05)),
            "ic90_alto": float(np.quantile(dif, 0.95)),
            "prob_mejora": float((dif > 0).mean())}


def recalibrar(season=SEASON):
    from src.elo import load_nfl_data

    pd.set_option("display.width", 200)
    print("── 1 · Panel de ausencias " + "─" * 46)
    fuentes = cargar_fuentes(list(range(2012, season + 1)))
    largo, juegos = tabla_larga(fuentes)

    print("── 2 · P(no juega | estatus) " + "─" * 43)
    tabla_p = estimar_prob_ausencia(largo, range(2012, TEST[1] + 1))
    print(tabla_p.round(3).to_string(index=False))
    probs = tabla_a_probs(tabla_p)
    tabla_p.to_csv(DATA_DIR / "prob_ausencia.csv", index=False)

    pan_auto, _ = agregar_panel(largo, juegos, probs, "auto")      # viernes
    pan_prev, _ = agregar_panel(largo, juegos, probs, "previa")    # martes
    panel = pan_auto.merge(
        pan_prev[["game_id", "team"] + [f"A_pre_{g}" for g in GRUPOS]].rename(
            columns={f"A_pre_{g}": f"A_prov_{g}" for g in GRUPOS}),
        on=["game_id", "team"], how="left")
    panel.to_csv(DATA_DIR / "ausencias_panel.csv", index=False)
    print(f"Guardado: ausencias_panel.csv ({len(panel)} equipo-partido)")

    print("── 3 · Calibración de δ (entrenamiento "
          f"{TRAIN[0]}-{TRAIN[1]}, ausencias reales) " + "─" * 8)
    df = load_nfl_data(None)
    H = {"real": ea.preparar_historico(df, panel, "real"),
         "pre": ea.preparar_historico(df, panel, "pre"),
         "prov": ea.preparar_historico(df, panel, "prov")}
    tr, te = _mask(H["real"], TRAIN), _mask(H["real"], TEST)

    cv = _cv_grupos(H["real"], TRAIN)
    print(cv.round(5).to_string(index=False))
    ll_todos = cv.loc[cv["config"] == "todos", "log_loss_cv"].iat[0]
    fuera = [c.replace("sin_", "") for c, ll in zip(cv["config"], cv["log_loss_cv"])
             if c != "todos" and ll <= ll_todos]
    fijos = {g: 0.0 for g in fuera}
    if fuera:
        print(f"Grupos que no mejoran en validación cruzada -> δ = 0: {', '.join(fuera)}")

    d_train = ea.calibrar_deltas(H["real"], tr, fijos=fijos)
    d_qb = ea.calibrar_deltas(H["real"], tr, grupos=("QB",),
                              fijos={g: 0.0 for g in GRUPOS if g != "QB"})
    ic = ea.bootstrap_deltas(H["real"], tr, n=300, grupos=tuple(g for g in GRUPOS if g not in fijos))
    print("δ (entrenamiento):", d_train)
    print("Intervalo bootstrap 90 %:\n", ic.round(1).to_string())

    print(f"── 4 · Prueba fuera de muestra {TEST[0]}-{TEST[1]} " + "─" * 30)
    y = H["pre"]["y"]
    p_base, _, _ = ea.correr_historico(H["pre"], {g: 0.0 for g in GRUPOS})
    p_veg = _p_vegas(H["pre"]["df"])
    modelos = {
        "Modelo 1 · Elo base": p_base,
        "Parche QB (δ=90)": ea.correr_historico(H["pre"], {"QB": 90.0})[1],
        "Solo QB (calibrado)": ea.correr_historico(H["pre"], d_qb)[1],
        "Modelo 2 · overlay · reporte viernes": ea.correr_historico(H["pre"], d_train)[1],
        "Modelo 2 · integrado · reporte viernes":
            ea.correr_historico(H["pre"], d_train, "integrado")[1],
        "Modelo 2 · overlay · reporte martes": ea.correr_historico(H["prov"], d_train)[1],
        "Modelo 2 · overlay · ausencias reales": ea.correr_historico(H["real"], d_train)[1],
    }
    filas = []
    for nombre, p in modelos.items():
        m = ea.metricas(y, p, te)
        m.update(_bootstrap_mejora(y[te], p_base[te], p[te]))
        filas.append({"modelo": nombre, "periodo": f"{TEST[0]}-{TEST[1]}", **m})
    veg = te & ~np.isnan(p_veg)
    mv = ea.metricas(y, p_veg, veg)
    filas.append({"modelo": "Las Vegas (moneyline)", "periodo": f"{TEST[0]}-{TEST[1]}", **mv})
    bt = pd.DataFrame(filas)
    print(bt[["modelo", "partidos", "brier", "log_loss", "aciertos", "mejora_brier",
              "ic90_bajo", "ic90_alto", "prob_mejora"]].round(4).to_string(index=False))

    b_ov = bt.loc[bt["modelo"] == "Modelo 2 · overlay · reporte viernes", "brier"].iat[0]
    b_in = bt.loc[bt["modelo"] == "Modelo 2 · integrado · reporte viernes", "brier"].iat[0]
    modo = "integrado" if b_in < b_ov else "overlay"
    print(f"Modo elegido: {modo} (Brier overlay {b_ov:.4f} · integrado {b_in:.4f})")

    print(f"── 5 · δ finales ({TRAIN[0]}-{TEST[1]}) " + "─" * 40)
    todo = _mask(H["real"], (TRAIN[0], TEST[1]))
    d_final = ea.calibrar_deltas(H["real"], todo, fijos=fijos)
    print("δ finales:", d_final)
    _, _, fin = ea.correr_historico(H["pre"], d_final, "integrado")
    pd.DataFrame({"team": list(fin), "elo_final": [round(v, 4) for v in fin.values()]}) \
      .sort_values("elo_final", ascending=False) \
      .to_csv(DATA_DIR / f"elo_final_ausencias_{TEST[1]}.csv", index=False)

    params = {
        "deltas": d_final,
        "modo": modo,
        "prob_ausencia": probs,
        "grupos_en_cero": fuera,
        "deltas_entrenamiento": d_train,
        "ic90_deltas_entrenamiento": {g: [round(float(ic.loc[g, "p05"]), 1),
                                          round(float(ic.loc[g, "p95"]), 1)] for g in ic.index},
        "entrenamiento": list(TRAIN),
        "prueba": list(TEST),
        "brier_prueba": {r.modelo: round(r.brier, 4) for r in bt.itertuples()},
        "nota": "δ = puntos Elo por titular ausente. Ver Notebooks/07_elo_ausencias.ipynb",
    }
    ea.guardar_parametros(params)

    # ── 2026 hasta hoy (fuera de muestra real) ──────────────────────────────
    bt26 = evaluar_temporada_actual(season, params)
    if bt26 is not None:
        bt = pd.concat([bt, bt26], ignore_index=True)
    bt.to_csv(DATA_DIR / "backtest_ausencias.csv", index=False)
    print("Guardado: backtest_ausencias.csv")
    return params, bt


def _modelos_temporada(season, params=None, zona=None):
    """games enriquecidos del Modelo 1 y resultado del Modelo 2 para `season`."""
    from src import weekly_content as wc

    schedule, final_elos = wc.cargar_datos(season)
    games_df, _ = wc.calcular_elo(schedule, final_elos)
    games = wc.enriquecer(games_df, schedule, zona or wc.ZONA_HORARIA)
    m2 = ea.modelo_ausencias_temporada(schedule, final_elos, season, params=params)
    m2["enriquecido"] = wc.enriquecer(m2["games"], schedule, zona or wc.ZONA_HORARIA)
    return games, m2


def evaluar_temporada_actual(season, params):
    games, m2 = _modelos_temporada(season, params)
    j = games[games["played"]]
    if j.empty:
        return None
    j2 = m2["games"].set_index(["home_team", "away_team", "week"]).loc[
        list(zip(j["home_team"], j["away_team"], j["week"]))]
    y = np.where(j["home_score"] == j["away_score"], 0.5,
                 (j["home_score"] > j["away_score"]).astype(float))
    filas = []
    for nombre, p in (("Modelo 1 · Elo base", j["p_home_pre"].to_numpy(float)),
                      (f"Modelo 2 · {m2['modo']} · reporte viernes",
                       j2["p_home_pre"].to_numpy(float))):
        filas.append({"modelo": nombre, "periodo": f"{season} sem. 1-{int(j['week'].max())}",
                      **ea.metricas(y, p)})
    pv = j["p_home_vegas"].to_numpy(float)
    ok = ~np.isnan(pv)
    filas.append({"modelo": "Las Vegas (moneyline)",
                  "periodo": f"{season} sem. 1-{int(j['week'].max())}", **ea.metricas(y[ok], pv[ok])})
    t = pd.DataFrame(filas)
    print(t.round(4).to_string(index=False))
    return t


# ══════════════════════════════════════════════════════════════════════════════
# Backfill y tabla
# ══════════════════════════════════════════════════════════════════════════════
def backfill(season=SEASON):
    """
    Congela en el libro las semanas YA JUGADAS que no tengan pronóstico, con el
    reporte final de lesiones de cada semana (origen = backfill).
    """
    games, m2 = _modelos_temporada(season)
    semanas = sorted(games.loc[games["played"], "week"].unique())
    if not semanas:
        print("No hay semanas jugadas.")
        return
    params = f"{m2['modo']}|{m2['params'].get('calibrado_en') or 'sin calibrar'}"
    n1 = libro_pron.registrar_pronosticos(games[games["played"]], "elo_base", semanas, season,
                                          "backfill", params="base")
    n2 = libro_pron.registrar_pronosticos(m2["enriquecido"][m2["enriquecido"]["played"]],
                                          "elo_ausencias", semanas, season, "backfill",
                                          params=params)
    print(f"Backfill semanas {semanas[0]}-{semanas[-1]}: +{len(n1)} Modelo 1 · +{len(n2)} Modelo 2 "
          f"→ {libro_pron.ruta_libro(season).relative_to(ROOT)}")
    tabla(season, games)


def tabla(season=SEASON, games=None):
    pd.set_option("display.width", 200)
    if games is None:
        games, _ = _modelos_temporada(season)
    libro = libro_pron.cargar_libro(season)
    t, ev = libro_pron.evaluar_modelos(libro, games)
    if t.empty:
        print("El libro no tiene partidos jugados con pronóstico de ambos modelos.")
        return
    res = libro_pron.resumen_comparacion(t)
    print(res[["nombre", "hasta_semana", "partidos", "record", "pct_aciertos", "brier",
               "log_loss", "vegas_brier"]].round(4).to_string(index=False))
    por_sem = t.pivot_table(index="semana", columns="modelo", values=["record", "brier"],
                            aggfunc="first")
    print(por_sem.to_string())
    des = libro_pron.desacuerdos(ev)
    if len(des):
        print(f"\nDistinto favorito en {len(des)} partidos:")
        print(des[["week", "away_team", "home_team", "favorito_elo_base", "favorito_elo_ausencias",
                   "ganador", "acerto"]].to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(description="Modelo 1 vs Modelo 2 (Elo + ausencias)")
    p.add_argument("--season", type=int, default=SEASON)
    p.add_argument("--recalibrar", action="store_true", help="Re-estima δ y P(ausencia)")
    p.add_argument("--backfill", action="store_true",
                   help="Congela en el libro las semanas ya jugadas")
    p.add_argument("--tabla", action="store_true", help="Comparación acumulada desde el libro")
    a = p.parse_args(argv)
    if not (a.recalibrar or a.backfill or a.tabla):
        p.print_help()
        return
    if a.recalibrar:
        recalibrar(a.season)
    if a.backfill:
        backfill(a.season)
    if a.tabla and not a.backfill:
        tabla(a.season)


if __name__ == "__main__":
    main()
