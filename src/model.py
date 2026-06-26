# src/model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import is_classifier
import inspect

# ── Constantes ────────────────────────────────────────────────────────────────

FEATURES = ['elo_diff', 'home_rest', 'away_rest', 'div_game', 'week_norm']
TARGET   = 'home_team_win'


# ── Función 1: construir features ─────────────────────────────────────────────

def build_features(history_df, schedules_df, home_advantage):
    """
    Une el ELO history con schedules y construye los features del modelo.

    Parameters
    ----------
    history_df     : pd.DataFrame — output de run_elo_system()
    schedules_df   : pd.DataFrame — output de load_nfl_data()
    home_advantage : float        — HOME_ADVANTAGE calibrado

    Returns
    -------
    pd.DataFrame con features listos para modelar
    """
    schedule_cols = [
        'season', 'week', 'game_type',
        'home_team', 'away_team',
        'home_rest', 'away_rest',
        'div_game', 'home_team_win'
    ]

    sched = schedules_df[schedule_cols].copy()

    # Unir ratings ELO pre-partido con schedules
    df = history_df[[
        'season', 'week', 'game_type',
        'home_team', 'away_team',
        'elo_home_pre', 'elo_away_pre',
    ]].merge(sched, on=['season', 'week', 'game_type', 'home_team', 'away_team'])

    # ── Construir features ────────────────────────────────────────────────────

    # Feature principal: diferencia de ELO incluyendo home advantage
    df['elo_diff'] = df['elo_home_pre'] - df['elo_away_pre'] + home_advantage

    # Semana normalizada: resuelve transición 16→17 juegos en 2021
    max_week = df.groupby('season')['week'].transform('max')
    df['week_norm'] = df['week'] / max_week

    # Solo temporada regular y sin missings en features
    df = df[df['game_type'] == 'REG'].copy()
    df = df.dropna(subset=FEATURES + [TARGET])

    # Down-weight temporada COVID
    df['sample_weight'] = df['season'].apply(
        lambda s: 0.3 if s == 2020 else 1.0
    )

    return df.reset_index(drop=True)


# ── Función 2: walk-forward validation ────────────────────────────────────────

def walk_forward_validation(df, test_seasons, feature_cols=FEATURES,
                            model=None, scale_features=None):
    """
    Walk-forward validation con ventana expandible.

    Acepta cualquier modelo compatible con scikit-learn.
    El StandardScaler se aplica automáticamente según el tipo de modelo,
    o puede controlarse manualmente con scale_features.

    Parameters
    ----------
    df            : pd.DataFrame — output de build_features()
    test_seasons  : list         — temporadas de evaluación
    feature_cols  : list         — features a usar
    model         : sklearn estimator — modelo a evaluar
                    Default: LogisticRegression(C=1.0)
    scale_features: bool | None  — si None, se infiere desde el tipo de modelo
                    True  → aplica StandardScaler (modelos lineales)
                    False → sin scaler (modelos basados en árboles)

    Returns
    -------
    results_df : pd.DataFrame — predicciones fold a fold
    metrics_df : pd.DataFrame — BS, Accuracy, Log-Loss por fold
    """
    # Modelo default
    if model is None:
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    # Inferir si necesita scaling según tipo de modelo
    TREE_MODELS = (GradientBoostingClassifier,)

    if scale_features is None:
        scale_features = not isinstance(model, TREE_MODELS)

    model_name = type(model).__name__
    print(f"Modelo          : {model_name}")
    print(f"Scale features  : {scale_features}")
    print(f"Features        : {feature_cols}")
    print("-" * 65)

    all_predictions = []
    fold_metrics    = []

    for test_season in test_seasons:

        train = df[df['season'] <  test_season].copy()
        test  = df[df['season'] == test_season].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        X_train = train[feature_cols]
        y_train = train[TARGET]
        w_train = train['sample_weight']
        X_test  = test[feature_cols]
        y_test  = test[TARGET]

        # Scaling opcional — siempre fiteado solo sobre training
        if scale_features:
            scaler     = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)
        else:
            X_train_sc = X_train
            X_test_sc  = X_test

        # Clonar modelo para evitar contaminación entre folds
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train_sc, y_train, sample_weight=w_train)

        p_home = fold_model.predict_proba(X_test_sc)[:, 1]

        test = test.copy()
        test['p_model'] = p_home
        test['fold']    = test_season
        all_predictions.append(test)

        bs  = brier_score_loss(y_test, p_home)
        ll  = log_loss(y_test, p_home)
        acc = accuracy_score(y_test, (p_home > 0.5).astype(int))

        fold_metrics.append({
            'test_season': test_season,
            'train_size' : len(train),
            'test_size'  : len(test),
            'brier_score': round(bs, 4),
            'log_loss'   : round(ll, 4),
            'accuracy'   : round(acc, 4),
        })

        print(
            f"Fold {test_season} | "
            f"Train={len(train):,} | "
            f"Test={len(test):,} | "
            f"BS={bs:.4f} | "
            f"Acc={acc:.1%}"
        )

    results_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)

    return results_df, metrics_df


# ── Función 3: modelo final ───────────────────────────────────────────────────

def train_final_model(df, feature_cols=FEATURES):
    """
    Entrena el modelo final sobre TODOS los datos disponibles.
    Este modelo predice la temporada 2026.

    Returns
    -------
    model   : LogisticRegression entrenado
    scaler  : StandardScaler fiteado
    coef_df : pd.DataFrame con coeficientes β ordenados por importancia
    """
    X = df[feature_cols]
    y = df[TARGET]
    w = df['sample_weight']

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_sc, y, sample_weight=w)

    coef_df = pd.DataFrame({
        'feature'    : feature_cols,
        'coefficient': model.coef_[0].round(4),
        'abs_coef'   : np.abs(model.coef_[0]).round(4),
    }).sort_values('abs_coef', ascending=False).reset_index(drop=True)

    return model, scaler, coef_df