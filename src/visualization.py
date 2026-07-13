# src/visualization.py

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
import nflreadpy as nfl

# ── Constantes ────────────────────────────────────────────────────────────────

AFC_COLOR  = '#013369'
NFC_COLOR  = '#D50A0A'
GOLD_COLOR = '#f5c518'

DIV_ORDER = [
    ['AFC_East', 'AFC_North', 'AFC_South', 'AFC_West'],
    ['NFC_East', 'NFC_North', 'NFC_South', 'NFC_West'],
]

DIV_LABELS = {
    'AFC_East' : 'AFC EAST',  'AFC_North': 'AFC NORTH',
    'AFC_South': 'AFC SOUTH', 'AFC_West' : 'AFC WEST',
    'NFC_East' : 'NFC EAST',  'NFC_North': 'NFC NORTH',
    'NFC_South': 'NFC SOUTH', 'NFC_West' : 'NFC WEST',
}


# En src/visualization.py — copiar estas constantes de simulation.py

TEAM_CONF = {
    'BAL': 'AFC', 'CLE': 'AFC', 'PIT': 'AFC', 'CIN': 'AFC',
    'HOU': 'AFC', 'IND': 'AFC', 'JAX': 'AFC', 'TEN': 'AFC',
    'BUF': 'AFC', 'MIA': 'AFC', 'NE' : 'AFC', 'NYJ': 'AFC',
    'KC' : 'AFC', 'LV' : 'AFC', 'LAC': 'AFC', 'DEN': 'AFC',
    'CHI': 'NFC', 'DET': 'NFC', 'GB' : 'NFC', 'MIN': 'NFC',
    'ATL': 'NFC', 'CAR': 'NFC', 'NO' : 'NFC', 'TB' : 'NFC',
    'DAL': 'NFC', 'NYG': 'NFC', 'PHI': 'NFC', 'WAS': 'NFC',
    'LA' : 'NFC', 'SEA': 'NFC', 'SF' : 'NFC', 'ARI': 'NFC',
}

# ── Paleta clara (paper) ──────────────────────────────────────────────────────
PAPER_BG   = '#FFFFFF'   # fondo blanco de la figura
CARD_LIGHT = '#F2F2F2'   # card gris claro (no-líderes)
TEXT_DARK  = '#1C1C1E'   # texto primario oscuro
TEXT_MUTED = '#6B6B6B'   # texto secundario
TEXT_FAINT = '#9A9A9A'   # etiquetas tenues
BAR_TRACK  = '#DDDDDD'   # riel de barra vacío
ACCENT     = '#E9AA3B'   # naranja de acento (reemplaza el dorado)
# AFC_COLOR y NFC_COLOR se mantienen — funcionan sobre blanco

# ── Helpers de logos ──────────────────────────────────────────────────────────

_logo_cache = {}
_logo_urls  = None

ABBR_MAP = {
    'LA' : 'LAR',
    'JAX': 'JAC',
}

def _load_logo_urls():
    """Carga URLs de logos desde nfl_data_py (solo una vez)."""
    global _logo_urls
    if _logo_urls is None:
        teams_desc = nfl.load_teams().to_pandas()
        _logo_urls = dict(zip(
            teams_desc['team_abbr'],
            teams_desc['team_logo_espn']
        ))
    return _logo_urls


def _get_logo(team, size=(45, 45)):
    """Descarga, redimensiona y cachea el logo de un equipo."""
    cache_key = (team, size)
    if cache_key in _logo_cache:
        return _logo_cache[cache_key]

    logo_urls = _load_logo_urls()
    espn_abbr = ABBR_MAP.get(team, team)
    url       = logo_urls.get(espn_abbr, logo_urls.get(team))

    if url is None:
        print(f"⚠️  No logo para {team}")
        return None
    try:
        r   = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert('RGBA')
        img = img.resize(size, Image.LANCZOS)
        _logo_cache[cache_key] = img
        return img
    except Exception as e:
        print(f"⚠️  Error descargando logo {team}: {e}")
        return None


def _add_logo(ax, team, x, y, zoom=0.50):
    """Coloca el logo de un equipo en las coordenadas (x, y) del eje."""
    img = _get_logo(team)
    if img is not None:
        imagebox = OffsetImage(np.array(img), zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y),
                            frameon=False, zorder=5,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)


# ── Viz 1: Standings por división ─────────────────────────────────────────────

def plot_regular_season_standings(results_df, bg_color=PAPER_BG,
                                  save_path=None):
    plt.close('all')

    fig, axes = plt.subplots(2, 4, figsize=(22, 13))
    fig.patch.set_facecolor(bg_color)
    fig.suptitle(
        'NFL 2026 · Projected Regular Season\n10,000 Monte Carlo simulations',
        color=TEXT_DARK, fontsize=19, fontweight='bold', y=0.99
    )

    for row, (conf, divs) in enumerate(zip(['AFC', 'NFC'], DIV_ORDER)):
        conf_color = AFC_COLOR if conf == 'AFC' else NFC_COLOR

        for col, div in enumerate(divs):
            ax = axes[row][col]
            ax.set_facecolor(bg_color)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')

            # Header
            ax.add_patch(FancyBboxPatch(
                (0, 8.8), 10, 1.1,
                boxstyle="round,pad=0.05",
                facecolor=conf_color, edgecolor='none'
            ))
            ax.text(5, 9.4, DIV_LABELS[div],
                    color='white', fontsize=14, fontweight='black',
                    ha='center', va='center')

            div_teams = (
                results_df[results_df['division'] == div]
                .sort_values('avg_wins', ascending=False)
                .reset_index(drop=True)
            )

            y_positions = [7.4, 5.3, 3.2, 1.1]
            card_h      = 1.7

            for i, row_data in div_teams.iterrows():
                team      = row_data['team']
                y         = y_positions[i]
                is_leader = (i == 0)

                # Fills
                face = conf_color if is_leader else CARD_LIGHT
                edge = ACCENT     if is_leader else '#D0D0D0'
                lw   = 2.0         if is_leader else 0.8

                # Colores de texto según el fondo de la card
                name_c  = 'white'    if is_leader else TEXT_DARK
                sub_c   = '#E6E6E6'  if is_leader else TEXT_MUTED
                pct_c   = ACCENT     if is_leader else TEXT_DARK
                lbl_c   = '#D8D8D8'  if is_leader else TEXT_FAINT
                pos_c   = ACCENT     if is_leader else TEXT_MUTED
                bar_c   = ACCENT     if is_leader else conf_color
                trk_c   = 'white'    if is_leader else BAR_TRACK
                trk_a   = 0.25       if is_leader else 1.0

                ax.add_patch(FancyBboxPatch(
                    (0.2, y), 9.6, card_h,
                    boxstyle="round,pad=0.05",
                    facecolor=face, edgecolor=edge, linewidth=lw
                ))

                ax.text(0.7, y + card_h/2, f'{i+1}',
                        color=pos_c, fontsize=16,
                        fontweight='black', va='center', ha='center')

                _add_logo(ax, team, 1.9, y + card_h/2)

                ax.text(3.0, y + card_h * 0.70, team,
                        color=name_c, fontsize=16,
                        fontweight='black', va='center')

                ax.text(3.0, y + card_h * 0.38,
                        f'{row_data["avg_wins"]:.1f} wins  ±{row_data["std_wins"]:.1f}',
                        color=sub_c, fontsize=10, va='center')

                # Barra de wins
                bar_x0, bar_w, bar_hr = 3.0, 5.5, 0.18
                bar_y  = y + card_h * 0.16
                fill_w = bar_w * (row_data['avg_wins'] / 17)

                ax.add_patch(FancyBboxPatch(
                    (bar_x0, bar_y), bar_w, bar_hr,
                    boxstyle="round,pad=0.02",
                    facecolor=trk_c, alpha=trk_a, edgecolor='none'
                ))
                ax.add_patch(FancyBboxPatch(
                    (bar_x0, bar_y), max(fill_w, 0.1), bar_hr,
                    boxstyle="round,pad=0.02",
                    facecolor=bar_c, edgecolor='none'
                ))

                ax.text(9.6, y + card_h * 0.70,
                        f'{row_data["pct_playoffs"]:.0f}%',
                        color=pct_c, fontsize=15, fontweight='bold',
                        va='center', ha='right')
                ax.text(9.6, y + card_h * 0.38, 'playoffs',
                        color=lbl_c, fontsize=8.5,
                        va='center', ha='right')

                if is_leader:
                    ax.text(9.6, y + card_h * 0.14,
                            f'div {row_data["pct_div_winner"]:.0f}%',
                            color=lbl_c, fontsize=8.5,
                            va='center', ha='right')

    # Leyenda
    legend_ax = fig.add_axes([0.01, 0.005, 0.98, 0.025])
    legend_ax.axis('off')
    legend_ax.set_facecolor(bg_color)
    legend_ax.legend(
        handles=[
            mpatches.Patch(facecolor=AFC_COLOR, label='AFC division leader'),
            mpatches.Patch(facecolor=NFC_COLOR, label='NFC division leader'),
            mpatches.Patch(facecolor=CARD_LIGHT, label='Other teams'),
            plt.Line2D([0], [0], color=ACCENT, linewidth=2.5,
                       label='Accent border = division leader'),
        ],
        loc='center', ncol=4, fontsize=11,
        framealpha=0, labelcolor=TEXT_DARK, handlelength=1.5
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor=bg_color)
        print(f"Guardado en {save_path}")

    plt.close(fig)
    return fig


# ── Viz 2: Wild Card Bracket ──────────────────────────────────────────────────

def get_most_likely_bracket(results_df):
    """
    Extrae el bracket más probable del Monte Carlo.
    Seeds 1-4: campeones de división por avg_wins.
    Seeds 5-7: wildcards por avg_wins.

    Returns
    -------
    dict: {'AFC': [s1,...,s7], 'NFC': [s1,...,s7]}
    """
    from src.simulation import CONFERENCES, DIVISIONS, TEAM_DIV

    div_winners = {}
    for div, teams in DIVISIONS.items():
        div_teams = results_df[results_df['team'].isin(teams)]
        winner    = div_teams.loc[div_teams['pct_div_winner'].idxmax(), 'team']
        div_winners[div] = winner

    bracket = {}
    for conf, all_teams in CONFERENCES.items():
        conf_divs    = [d for d in div_winners if d.startswith(conf)]
        conf_winners = [div_winners[d] for d in conf_divs]

        seeds_14 = sorted(
            conf_winners,
            key=lambda t: results_df.loc[
                results_df['team'] == t, 'avg_wins'
            ].values[0],
            reverse=True
        )

        non_winners = results_df[
            results_df['conf'].eq(conf) &
            ~results_df['team'].isin(conf_winners)
        ].nlargest(3, 'avg_wins')['team'].tolist()

        bracket[conf] = seeds_14 + non_winners

    return bracket


def _draw_wc_card(ax, x0, y0, w, h, seed, team, meta, pct,
                  is_div, color, show_vs=False):
    face = color if is_div else CARD_LIGHT
    edge = ACCENT if is_div else color

    # Texto según fondo
    seed_c = 'white'   if is_div else TEXT_DARK
    name_c = 'white'   if is_div else TEXT_DARK
    meta_c = '#E6E6E6' if is_div else TEXT_MUTED
    pct_c  = 'white'   if is_div else TEXT_DARK
    lbl_c  = '#D8D8D8' if is_div else TEXT_FAINT

    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.04",
        facecolor=face, edgecolor=edge, linewidth=1.0
    ))

    ax.text(x0 + 0.45, y0 + h/2, str(seed),
            color=seed_c, fontsize=15, fontweight='black',
            va='center', ha='center')

    _add_logo(ax, team, x0 + 1.35, y0 + h/2)

    ax.text(x0 + 2.3, y0 + h * 0.68, team,
            color=name_c, fontsize=17, fontweight='black', va='center')
    ax.text(x0 + 2.3, y0 + h * 0.28, meta,
            color=meta_c, fontsize=9, va='center')

    ax.text(x0 + w - 0.15, y0 + h * 0.68, f'{pct:.1f}%',
            color=pct_c, fontsize=13, fontweight='bold',
            va='center', ha='right')
    ax.text(x0 + w - 0.15, y0 + h * 0.28, 'conf champ',
            color=lbl_c, fontsize=8, va='center', ha='right')

    if show_vs:
        ax.text(x0 + w/2, y0 - 0.08, '·  vs  ·',
                color='#BBBBBB', fontsize=9,
                ha='center', va='center', style='italic')


def plot_wildcard_bracket(results_df, bg_color=PAPER_BG, save_path=None):
    plt.close('all')

    bracket = get_most_likely_bracket(results_df)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor(bg_color)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.06)
    fig.suptitle(
        'NFL 2026 · Wild Card Preview\nMost likely bracket · 10,000 Monte Carlo simulations',
        color=TEXT_DARK, fontsize=17, fontweight='bold', y=0.98
    )

    groups_y = [8.1, 6.1, 4.1, 2.1]
    card_h   = 0.78

    for ax, conf, color in [(axes[0], 'AFC', AFC_COLOR), (axes[1], 'NFC', NFC_COLOR)]:
        ax.set_facecolor(bg_color)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        seeds = bracket[conf]
        ax.text(5, 9.7, conf, color=color, fontsize=24,
                fontweight='black', ha='center', va='center')

        # Seed 1 BYE
        s    = seeds[0]
        info = results_df[results_df['team'] == s].iloc[0]
        y    = groups_y[0]
        _draw_wc_card(ax, 0.3, y - card_h/2, 9.4, card_h,
                      seed=1, team=s,
                      meta=f"{info['avg_wins']:.1f} wins avg · {info['division']}",
                      pct=info['pct_conf_champ'], is_div=True, color=color)
        ax.text(9.3, y, 'BYE', color='white', fontsize=11,
                fontweight='bold', va='center', ha='right')

        for gi, (hi, lo) in enumerate([(1, 6), (2, 5), (3, 4)]):
            y       = groups_y[gi + 1]
            info_hi = results_df[results_df['team'] == seeds[hi]].iloc[0]
            info_lo = results_df[results_df['team'] == seeds[lo]].iloc[0]

            _draw_wc_card(ax, 0.3, y + 0.04, 9.4, card_h,
                          seed=hi+1, team=seeds[hi],
                          meta=f"{info_hi['avg_wins']:.1f} wins · {info_hi['division']}",
                          pct=info_hi['pct_conf_champ'],
                          is_div=True, color=color, show_vs=True)
            _draw_wc_card(ax, 0.3, y - card_h - 0.04, 9.4, card_h,
                          seed=lo+1, team=seeds[lo],
                          meta=f"{info_lo['avg_wins']:.1f} wins · WILD CARD",
                          pct=info_lo['pct_conf_champ'],
                          is_div=False, color=color)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
        print(f"Guardado en {save_path}")

    plt.close(fig)
    return fig

def plot_team_schedule(team, game_results_df,
                       bg_color='#1c1c1e', save_path=None):
    """
    Tabla visual del calendario 2026 de un equipo con
    probabilidades de victoria por partido.

    Parameters
    ----------
    team            : str          — abreviación del equipo (ej. 'KC')
    game_results_df : pd.DataFrame — output de run_monte_carlo()
    bg_color        : str          — color de fondo
    save_path       : str | None
    """
    # ── Preparar datos del equipo ─────────────────────────────────────────────
    mask = (
        (game_results_df['home_team'] == team) |
        (game_results_df['away_team'] == team)
    )
    games = game_results_df[mask].copy().reset_index(drop=True)

    games['is_home']  = games['home_team'] == team
    games['opponent'] = games.apply(
        lambda r: r['away_team'] if r['is_home'] else r['home_team'],
        axis=1
    )
    games['p_win'] = games.apply(
        lambda r: r['p_home_win'] if r['is_home'] else r['p_away_win'],
        axis=1
    )
    games['sim_wins'] = games.apply(
        lambda r: r['home_wins'] if r['is_home'] else r['away_wins'],
        axis=1
    )

    n_games   = len(games)
    conf_team = TEAM_CONF.get(team, '')
    color     = AFC_COLOR if conf_team == 'AFC' else NFC_COLOR
    card_bg   = '#2c2c2e'

    # ── Figura ─────────────────────────────────────────────────────────────────
    fig_h = 1.2 + n_games * 0.72
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # ── Header ────────────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0, fig_h - 1.0), 12, 1.0,
        boxstyle="square,pad=0",
        facecolor=color, edgecolor='none'
    ))
    _add_logo(ax, team, 0.7, fig_h - 0.48, zoom=0.45)
    ax.text(1.6, fig_h - 0.32, team,
            color='white', fontsize=22, fontweight='black', va='center')
    ax.text(1.6, fig_h - 0.72, 'Calendario Temporada Regular 2026',
            color='white', alpha=0.7, fontsize=11, va='center')

    # Avg wins proyectado
    avg_w = games['p_win'].sum()
    ax.text(11.7, fig_h - 0.38, f'{avg_w:.1f}',
            color=GOLD_COLOR, fontsize=28, fontweight='black',
            va='center', ha='right')
    ax.text(11.7, fig_h - 0.75, 'wins esperados',
            color='white', alpha=0.6, fontsize=9,
            va='center', ha='right')

    # ── Columnas header ───────────────────────────────────────────────────────
    y_col = fig_h - 1.22
    for x, label, align in [
        (0.5,  'WK',      'center'),
        (1.7,  'RIVAL',   'left'),
        (5.2,  'LOCAL',   'center'),
        (6.8,  'P(WIN)',  'center'),
        (9.5,  'SIMS GANADAS', 'center'),
    ]:
        ax.text(x, y_col, label, color='#666666',
                fontsize=8, fontweight='bold',
                va='center', ha=align,
                transform=ax.transData)

    ax.add_patch(plt.Rectangle(
        (0, y_col - 0.18), 12, 0.02,
        facecolor='#333333', edgecolor='none'
    ))

    # ── Filas por partido ─────────────────────────────────────────────────────
    y = fig_h - 1.65
    row_h = 0.65

    for _, g in games.iterrows():
        opp      = g['opponent']
        p_win    = g['p_win']
        is_home  = g['is_home']
        sim_wins = int(g['sim_wins'])
        week     = int(g['week'])

        # Fondo de fila alternada
        row_color = card_bg if int(week) % 2 == 0 else bg_color
        ax.add_patch(FancyBboxPatch(
            (0.1, y - row_h/2 + 0.05), 11.8, row_h - 0.05,
            boxstyle="round,pad=0.04",
            facecolor=row_color, edgecolor='none'
        ))

        # Semana
        ax.text(0.5, y, str(week), color='#888888',
                fontsize=11, fontweight='bold',
                va='center', ha='center')

        # Logo rival
        _add_logo(ax, opp, 2.0, y, zoom=0.30)

        # Nombre rival
        ax.text(2.9, y, opp, color='white',
                fontsize=13, fontweight='black', va='center')

        # Local/Visitante
        loc_text  = '🏠 LOCAL' if is_home else '✈ VISIT'
        loc_color = '#a6e3a1' if is_home else '#aaaaaa'
        ax.text(5.2, y, loc_text, color=loc_color,
                fontsize=9, va='center', ha='center')

        # Barra de probabilidad
        bar_x   = 5.9
        bar_w   = 2.6
        bar_h_r = 0.25
        bar_y   = y - bar_h_r/2

        ax.add_patch(FancyBboxPatch(
            (bar_x, bar_y), bar_w, bar_h_r,
            boxstyle="round,pad=0.02",
            facecolor='#383838', edgecolor='none'
        ))

        fill_color = '#a6e3a1' if p_win >= 0.55 else \
                     '#f38ba8' if p_win < 0.45 else \
                     GOLD_COLOR

        ax.add_patch(FancyBboxPatch(
            (bar_x, bar_y), bar_w * p_win, bar_h_r,
            boxstyle="round,pad=0.02",
            facecolor=fill_color, alpha=0.9, edgecolor='none'
        ))

        ax.text(6.7, y, f'{p_win:.0%}', color='white',
                fontsize=12, fontweight='bold',
                va='center', ha='center')

        # Simulaciones ganadas (barra + número)
        sim_bar_x = 8.2
        sim_bar_w = 2.8
        n_sims_total = int(g['home_wins']) + int(g['away_wins'])

        ax.add_patch(FancyBboxPatch(
            (sim_bar_x, bar_y), sim_bar_w, bar_h_r,
            boxstyle="round,pad=0.02",
            facecolor='#383838', edgecolor='none'
        ))
        ax.add_patch(FancyBboxPatch(
            (sim_bar_x, bar_y), sim_bar_w * (sim_wins/n_sims_total), bar_h_r,
            boxstyle="round,pad=0.02",
            facecolor=fill_color, alpha=0.5, edgecolor='none'
        ))
        ax.text(11.3, y, f'{sim_wins:,} / {n_sims_total:,}',
                color='#aaaaaa', fontsize=9,
                va='center', ha='right')

        y -= row_h

    # ── Leyenda de colores ────────────────────────────────────────────────────
    for x, color_l, label in [
        (1.0,  '#a6e3a1', 'Favorito (>55%)'),
        (4.5,  GOLD_COLOR,'Parejo (45–55%)'),
        (8.0,  '#f38ba8', 'Underdog (<45%)'),
    ]:
        ax.add_patch(plt.Rectangle(
            (x, 0.18), 0.3, 0.25,
            facecolor=color_l, edgecolor='none'
        ))
        ax.text(x + 0.45, 0.30, label, color='#777777',
                fontsize=8, va='center')

    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=bg_color)
        print(f'Guardado: {save_path}')

    plt.close(fig)