"""
clutch_performance.py
Rendimiento en situaciones cerradas (4Q + OT, diferencia ≤ 7 puntos).
  · Equipo específico → desglose detallado clutch vs temporada + ranking
  · Enter            → scatter 32 equipos: EPA ofensivo vs defensivo clutch
NFL 2025
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SEASON  = None   # None = auto-detectar última temporada

BG     = "#0f1115"
CARD   = "#151924"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
ACCENT = "#2d6cdf"
DPI    = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

MIN_PLAYS      = 20
MIN_PLAYS_TM   = 10
SCORE_DIFF_MAX = 7


def load_logo(team, base_zoom=0.07, fade=1.0):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path).astype(float)
        if img.max() > 1.0:
            img = img / 255.0
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        zoom = base_zoom / HARD_PENALTY[team] if team in HARD_PENALTY else \
               base_zoom / np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
        if fade < 1.0:
            if img.shape[2] == 4:
                img = img.copy(); img[..., 3] *= fade
            else:
                img = np.dstack([img, np.full(img.shape[:2], fade)])
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


# ── INPUT ─────────────────────────────────────────────────────────────────────
team = input("Equipo (siglas, ej: KC) o Enter para scatter 32 equipos: ").strip().upper()
modo = "equipo" if team else "grid"

# ── CARGA ─────────────────────────────────────────────────────────────────────
pbp, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(pbp):,} jugadas REG")
pbp["epa"]               = pd.to_numeric(pbp["epa"],               errors="coerce")
pbp["score_differential"] = pd.to_numeric(pbp["score_differential"], errors="coerce")

plays = pbp[
    pbp["play_type"].isin(["pass", "run"]) &
    pbp["epa"].notna() &
    pbp["score_differential"].notna()
].copy()

off_full_all = plays.groupby("posteam")["epa"].mean()
def_full_all = plays.groupby("defteam")["epa"].mean()

clutch = plays[
    plays["qtr"].isin([4, 5]) &
    plays["score_differential"].abs().le(SCORE_DIFF_MAX)
].copy()
print(f"Jugadas clutch: {len(clutch):,}")


# ══════════════════════════════════════════════════════════════════════════════
# MODO EQUIPO — desglose detallado
# ══════════════════════════════════════════════════════════════════════════════
if modo == "equipo":

    if team not in plays["posteam"].values:
        raise SystemExit(f"No se encontraron jugadas para {team}.")

    # ── Stats del equipo ───────────────────────────────────────────────────────
    def epa_stats(df, team_col, val, play_type=None):
        s = df[df[team_col] == val]
        if play_type:
            s = s[s["play_type"] == play_type]
        s = s["epa"].dropna()
        return (s.mean(), len(s)) if len(s) >= MIN_PLAYS_TM else (np.nan, len(s))

    # Offense
    off_season_epa, _ = epa_stats(plays,  "posteam", team)
    off_clutch_epa, _ = epa_stats(clutch, "posteam", team)
    off_pass_season,_ = epa_stats(plays,  "posteam", team, "pass")
    off_pass_clutch,_ = epa_stats(clutch, "posteam", team, "pass")
    off_run_season, _ = epa_stats(plays,  "posteam", team, "run")
    off_run_clutch, _ = epa_stats(clutch, "posteam", team, "run")

    # Defense
    def_season_epa, _ = epa_stats(plays,  "defteam", team)
    def_clutch_epa, _ = epa_stats(clutch, "defteam", team)
    def_pass_season,_ = epa_stats(plays,  "defteam", team, "pass")
    def_pass_clutch,_ = epa_stats(clutch, "defteam", team, "pass")
    def_run_season, _ = epa_stats(plays,  "defteam", team, "run")
    def_run_clutch, _ = epa_stats(clutch, "defteam", team, "run")

    # ── Rankings ──────────────────────────────────────────────────────────────
    off_clutch_all = clutch.groupby("posteam")["epa"].mean()
    def_clutch_all = clutch.groupby("defteam")["epa"].mean()

    n_off = off_clutch_all.notna().sum()
    n_def = def_clutch_all.notna().sum()

    off_rank = int(off_clutch_all.rank(ascending=False)[team]) if team in off_clutch_all.index else None
    def_rank = int(def_clutch_all.rank(ascending=True)[team])  if team in def_clutch_all.index else None

    lg_off_clutch = off_clutch_all.mean()
    lg_def_clutch = def_clutch_all.mean()

    # ── FIGURA ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                             hspace=0.3, wspace=0.44,
                             left=0.06, right=0.97,
                             top=0.86, bottom=0.07)

    colors_off = ["#27ae60", "#2d6cdf"]   # clutch, season
    colors_def = ["#27ae60", "#2d6cdf"]

    def draw_panel(ax, labels, clutch_vals, season_vals, lg_clutch, title, xlabel, higher_better=True):
        y = np.arange(len(labels))
        h = 0.35

        # Determina colores por desviación vs liga
        c_clutch = []
        for v in clutch_vals:
            if np.isnan(v):
                c_clutch.append(GRID)
            else:
                delta = (v - lg_clutch) * (1 if higher_better else -1)
                c_clutch.append("#27ae60" if delta >= 0.02 else ("#d84a4a" if delta <= -0.02 else "#ffd166"))

        c_season = ["#4a5a7a"] * len(labels)

        ax.barh(y + h/2, clutch_vals, height=h, color=c_clutch, zorder=3, label="Clutch")
        ax.barh(y - h/2, season_vals, height=h, color=c_season,  zorder=3, label="Temporada", alpha=0.7)

        max_abs = max((abs(v) for v in list(clutch_vals) + list(season_vals) if not np.isnan(v)), default=0.2)

        for i, (cv, sv) in enumerate(zip(clutch_vals, season_vals)):
            for val, row in [(cv, y[i] + h/2), (sv, y[i] - h/2)]:
                if not np.isnan(val):
                    sign = "+" if val >= 0 else ""
                    inside = abs(val) >= max_abs * 0.25
                    if inside:
                        ax.text(val / 2, row, f"{sign}{val:.3f}",
                                ha="center", va="center", color="#0a0e13",
                                fontsize=7.5, fontweight="bold", zorder=5)
                    else:
                        offset = max_abs * 0.03
                        ax.text(val + (offset if val >= 0 else -offset), row,
                                f"{sign}{val:.3f}",
                                ha="left" if val >= 0 else "right", va="center",
                                color=FG, fontsize=7.5, fontweight="bold", zorder=5)

            # Delta clutch vs season
            if not np.isnan(cv) and not np.isnan(sv):
                delta = cv - sv
                sign = "+" if delta >= 0 else ""
                col = "#27ae60" if (delta * (1 if higher_better else -1)) > 0 else "#d84a4a"
                ax.text(max_abs * 1.12, y[i], f"Δ {sign}{delta:.3f}",
                        ha="left", va="center", color=col,
                        fontsize=7, fontweight="bold", zorder=5)

        ax.axvline(0, color=FG, linewidth=0.7, alpha=0.4, zorder=2)
        ax.axvline(lg_clutch, color="#ffd166", linewidth=1.0, alpha=0.6,
                   linestyle="--", zorder=2, label="Media liga (clutch)")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=FG, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.4, zorder=1)
        ax.set_xlabel(xlabel, color=FG, fontsize=8.5)
        ax.set_title(title, color=FG, fontsize=10, pad=6, fontweight="bold", loc="left")
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=FG, length=0)
        ax.set_xlim(-(max_abs * 1.3), max_abs * 1.45)

        ax.legend(fontsize=7, labelcolor=FG, facecolor=CARD,
                  edgecolor=GRID, framealpha=0.6, loc="lower right")

    off_labels   = ["General", "Pase", "Carrera"]
    off_clutch_v = [off_clutch_epa, off_pass_clutch, off_run_clutch]
    off_season_v = [off_season_epa, off_pass_season, off_run_season]
    def_labels   = ["General", "Pase", "Carrera"]
    def_clutch_v = [def_clutch_epa, def_pass_clutch, def_run_clutch]
    def_season_v = [def_season_epa, def_pass_season, def_run_season]

    r_off = f"#{off_rank}/{n_off}" if off_rank else "n/d"
    r_def = f"#{def_rank}/{n_def}" if def_rank else "n/d"

    draw_panel(fig.add_subplot(gs[0]),
               off_labels, off_clutch_v, off_season_v,
               lg_off_clutch,
               f"Ataque Clutch  {team}  ({r_off})",
               "EPA / jugada  (barra verde = clutch, azul = temporada)",
               higher_better=True)

    draw_panel(fig.add_subplot(gs[1]),
               def_labels, def_clutch_v, def_season_v,
               lg_def_clutch,
               f"Defensa Clutch  {team}  ({r_def})",
               "EPA permitido  (verde = mejor que media liga clutch)",
               higher_better=False)

    # Logo
    logo = load_logo(team, base_zoom=0.095)
    if logo:
        lax = fig.add_axes([0.03, 0.912, 0.055, 0.075])
        lax.imshow(logo.get_data())
        lax.axis("off")

    fig.text(0.5, 0.975,
             f"{team}  |  Rendimiento Clutch  |  NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.952,
             f"Clutch = 4Q + prórroga con diferencia ≤ {SCORE_DIFF_MAX} pts  ·  Δ = diferencia clutch vs temporada  ·  --- = media liga clutch",
             ha="center", va="top", fontsize=8, color="#888", fontstyle="italic")
    fig.text(0.01, 0.008,
             f"Fuente: nflverse PBP  |  NFL {SEASON}  |  Mín {MIN_PLAYS_TM} jugadas",
             ha="left", va="bottom", fontsize=7, color="#555", fontstyle="italic")
    fig.text(0.99, 0.008, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888", alpha=0.8, fontstyle="italic")

    outfile = f"clutch_performance_{team}_{SEASON}.png"
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {outfile}")


# ══════════════════════════════════════════════════════════════════════════════
# MODO GRID — scatter 32 equipos
# ══════════════════════════════════════════════════════════════════════════════
else:

    off_clutch_agg = clutch.groupby("posteam").agg(off_epa=("epa","mean"), off_n=("epa","count"))
    def_clutch_agg  = clutch.groupby("defteam").agg(def_epa=("epa","mean"), def_n=("epa","count"))

    off_clutch_agg = off_clutch_agg[off_clutch_agg["off_n"] >= MIN_PLAYS]
    def_clutch_agg  = def_clutch_agg[def_clutch_agg["def_n"]  >= MIN_PLAYS]

    df = off_clutch_agg.join(def_clutch_agg, how="inner")
    df = df.join(off_full_all.rename("off_full"), how="left")
    df = df.join(def_full_all.rename("def_full"), how="left")
    df["off_delta"] = df["off_epa"] - df["off_full"]
    df["def_delta"]  = df["def_epa"]  - df["def_full"]
    df = df.reset_index().rename(columns={"posteam": "team"})
    print(f"Equipos con suficientes jugadas clutch: {len(df)}")

    fig, ax = plt.subplots(figsize=(13, 11), facecolor=BG)
    ax.set_facecolor(BG)

    x = df["off_epa"].values
    y = df["def_epa"].values

    x_center = np.nanmean(x)
    y_center = np.nanmean(y)

    # Límites explícitos — AnnotationBbox no actualiza autoscale
    x_margin = (np.nanmax(x) - np.nanmin(x)) * 0.18
    y_margin = (np.nanmax(y) - np.nanmin(y)) * 0.18
    x_min = np.nanmin(x) - x_margin
    x_max = np.nanmax(x) + x_margin
    y_min = np.nanmin(y) - y_margin
    y_max = np.nanmax(y) + y_margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Cuadrante verde (inferior-derecho = completos)
    ax.fill_betweenx([y_min, y_center], x_center, x_max,
                     color="#06d6a0", alpha=0.04, zorder=0)

    ax.axvline(x_center, color=GRID, linewidth=0.8, alpha=0.5, zorder=1)
    ax.axhline(y_center, color=GRID, linewidth=0.8, alpha=0.5, zorder=1)

    quad_kw = dict(fontsize=8, ha="center", va="center", fontstyle="italic", zorder=1)
    x_pad = (x_max - x_min) * 0.22
    y_pad = (y_max - y_min) * 0.22
    ax.text(x_center + x_pad, y_center - y_pad, "COMPLETOS\n(ataque + def. clutch)", color="#1a6b3a", **quad_kw)
    ax.text(x_center - x_pad, y_center - y_pad, "BUENOS DEF.\nmal ataque clutch",    color="#555",    **quad_kw)
    ax.text(x_center + x_pad, y_center + y_pad, "BUENOS ATQ.\nmala def. clutch",     color="#555",    **quad_kw)
    ax.text(x_center - x_pad, y_center + y_pad, "VULNERABLES\n(ambos lados)",        color="#7a2020", **quad_kw)

    for _, row in df.iterrows():
        tm   = row["team"]
        xi   = float(row["off_epa"])
        yi   = float(row["def_epa"])
        doff = float(row["off_delta"]) if not pd.isna(row["off_delta"]) else np.nan
        ddef = float(row["def_delta"]) if not pd.isna(row["def_delta"]) else np.nan
        if np.isnan(xi) or np.isnan(yi):
            continue
        if not (np.isnan(doff) or np.isnan(ddef)):
            ax.annotate("", xy=(xi + doff * 0.45, yi + ddef * 0.45), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.1, alpha=0.7),
                        zorder=2)
        logo = load_logo(tm, base_zoom=0.065)
        if logo:
            ab = AnnotationBbox(logo, (xi, yi), frameon=False, zorder=3, pad=0)
            ax.add_artist(ab)

    ax.set_xlabel("EPA ofensivo en clutch  (+ = mejor ataque)", color=FG, fontsize=10)
    ax.set_ylabel("EPA defensivo en clutch  (- = mejor defensa)", color=FG, fontsize=10)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.4, zorder=0)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=FG, labelsize=8)

    ax.annotate("", xy=(0.15, 0.04), xytext=(0.08, 0.04),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.2))
    ax.text(0.16, 0.04, "Flecha dorada = dirección del cambio\nvs rendimiento de temporada",
            color="#ffd166", fontsize=6.5, va="center", transform=ax.transAxes,
            alpha=0.8)

    fig.text(0.5, 0.975,
             f"Rendimiento Clutch  |  4Q + Prórroga, diferencia ≤ {SCORE_DIFF_MAX} puntos  |  NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.952,
             "Cuadrante inferior-derecho = equipos completos en situaciones críticas",
             ha="center", va="top", fontsize=8, color="#888", fontstyle="italic")
    fig.text(0.01, 0.008, f"Fuente: nflverse PBP  |  NFL {SEASON}  |  Mín {MIN_PLAYS} jugadas clutch",
             ha="left", va="bottom", fontsize=7, color="#555", fontstyle="italic")
    fig.text(0.99, 0.008, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888", alpha=0.8, fontstyle="italic")

    outfile = f"clutch_performance_{SEASON}.png"
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {outfile}")
