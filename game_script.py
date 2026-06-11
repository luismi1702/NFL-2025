"""
game_script.py
Cómo rinde un equipo según el contexto del partido.
Sin input → scatter 32 equipos (EPA liderando vs perdiendo).
Con input → 4 paneles: EPA por marcador · Pass rate vs liga · 3er down · 4th down.
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SEASON    = 2025
MIN_PLAYS = 25
URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

BG     = "#0f1115"
CARD   = "#151924"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
ACCENT = "#2d6cdf"
DPI    = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

COL_GO   = "#06d6a0"
COL_FG   = "#2d6cdf"
COL_PUNT = "#e07a5f"

# 5 bins para EPA / Pass rate / 3er down
BINS5 = [
    ("Ganando\n+8",    8,     9999),
    ("Ganando\n1–7",   1,     7),
    ("Empatado",       0,     0),
    ("Perdiendo\n1–7", -7,    -1),
    ("Perdiendo\n+8",  -9999, -8),
]
BIN5_KEYS  = [b[0] for b in BINS5]
BIN5_XLBLS = [k.replace("\n", " ") for k in BIN5_KEYS]

# 3 contextos para 4th down (muestra más pequeña)
BINS3 = [
    ("Ganando",   1,     9999),
    ("Empatado",  0,     0),
    ("Perdiendo", -9999, -1),
]
BIN3_KEYS = [b[0] for b in BINS3]


def load_logo(team, base_zoom=0.033):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        zoom = base_zoom / HARD_PENALTY[team] if team in HARD_PENALTY else \
               base_zoom / np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def assign_bin(sd, bins):
    for label, lo, hi in bins:
        if lo <= sd <= hi:
            return label
    return None


def style_ax(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=FG)
    plt.setp(ax.get_xticklabels(), color=FG)
    plt.setp(ax.get_yticklabels(), color=FG)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)


def bar_label(ax, i, val, fmt, v_abs, above=True):
    off = v_abs * 0.06
    ax.text(i, val + (off if above else -off), fmt,
            ha="center", va="bottom" if above else "top",
            color=FG, fontsize=8.5, fontweight="bold", zorder=3)


# ── INPUT ──────────────────────────────────────────────────────────────────────
team = input("Equipo (Enter=scatter 32 equipos): ").strip().upper()
modo = "equipo" if team else "grid"

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
raw = pd.read_csv(URL, low_memory=False, compression="infer")
for col in ["epa", "score_differential", "down"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# Jugadas ofensivas (pase/carrera) con EPA
plays = raw[
    raw["play_type"].isin(["pass", "run"]) &
    raw["epa"].notna() &
    raw["score_differential"].notna() &
    raw["posteam"].notna()
].copy()
plays["bin5"]    = plays["score_differential"].apply(lambda s: assign_bin(s, BINS5))
plays["bin3"]    = plays["score_differential"].apply(lambda s: assign_bin(s, BINS3))
plays["is_pass"] = (plays["play_type"] == "pass").astype(int)
plays            = plays[plays["bin5"].notna()]

# Jugadas de 4th down para decisiones
fourth = raw[
    (raw["down"] == 4) &
    raw["play_type"].isin(["pass", "run", "field_goal", "punt"]) &
    raw["score_differential"].notna() &
    raw["posteam"].notna()
].copy()
fourth["bin3"]      = fourth["score_differential"].apply(lambda s: assign_bin(s, BINS3))
fourth["decision"]  = fourth["play_type"].map(
    {"pass": "go", "run": "go", "field_goal": "fg", "punt": "punt"}
)
fourth = fourth[fourth["bin3"].notna()]

print(f"Jugadas: {len(plays):,}  |  4th down: {len(fourth):,}")

# ── MODO GRID ──────────────────────────────────────────────────────────────────
if modo == "grid":
    agg = plays.groupby(["posteam", "bin5"])["epa"].mean().unstack(fill_value=np.nan)

    tdf = pd.DataFrame(index=agg.index)
    lead = [c for c in ["Ganando\n+8", "Ganando\n1–7"] if c in agg.columns]
    trail = [c for c in ["Perdiendo\n+8", "Perdiendo\n1–7"] if c in agg.columns]
    tdf["epa_leading"]  = agg[lead].mean(axis=1)
    tdf["epa_trailing"] = agg[trail].mean(axis=1)
    tdf = tdf.dropna()

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    style_ax(ax)
    ax.grid(axis="both", color=GRID, linewidth=0.5, alpha=0.08, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_vals = tdf["epa_trailing"].values
    y_vals = tdf["epa_leading"].values
    x_pad  = (x_vals.max() - x_vals.min()) * 0.16
    y_pad  = (y_vals.max() - y_vals.min()) * 0.16
    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()

    diag_lo = max(x_lo, y_lo)
    diag_hi = min(x_hi, y_hi)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
            color=GRID, linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    ax.axvline(0, color=GRID, linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)
    ax.axhline(0, color=GRID, linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)

    xm = (x_hi - x_lo) * 0.025
    ym = (y_hi - y_lo) * 0.025
    q_kw = dict(fontsize=8.5, alpha=0.25, color=FG, fontstyle="italic")
    ax.text(x_hi - xm, y_hi - ym, "Élite en todo",              ha="right", va="top",    **q_kw)
    ax.text(x_lo + xm, y_hi - ym, "Solo brillan\ncon ventaja",  ha="left",  va="top",    **q_kw)
    ax.text(x_hi - xm, y_lo + ym, "Remontadores\nnatos",        ha="right", va="bottom", **q_kw)
    ax.text(x_lo + xm, y_lo + ym, "Necesita\nmejorar",          ha="left",  va="bottom", **q_kw)

    for team_id, row in tdf.iterrows():
        x, y = row["epa_trailing"], row["epa_leading"]
        logo = load_logo(team_id)
        if logo:
            ax.add_artist(AnnotationBbox(logo, (x, y), frameon=False,
                                         zorder=3, box_alignment=(0.5, 0.5)))
        else:
            ax.scatter(x, y, s=80, color="#888", zorder=3)
            ax.text(x, y, team_id, ha="center", va="center",
                    fontsize=7, color=FG, zorder=4)

    if diag_lo < diag_hi:
        mid = (diag_lo + diag_hi) / 2
        ax.text(mid, mid + (y_hi - y_lo) * 0.03,
                "igual rendimiento en ambos contextos",
                ha="center", va="bottom", fontsize=7, color=GRID,
                fontstyle="italic", rotation=38, rotation_mode="anchor")

    ax.set_xlabel("EPA/play  cuando va perdiendo", color=FG, fontsize=11, labelpad=7)
    ax.set_ylabel("EPA/play  cuando va ganando",   color=FG, fontsize=11, labelpad=7)

    fig.text(0.5, 0.97,
             f"Game Script  ·  EPA liderando vs perdiendo  ·  NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.925,
             "Por encima de la diagonal = rinden mejor con ventaja  ·  "
             "Por debajo = mejoran cuando van perdiendo",
             ha="center", va="top", fontsize=8.5, color="#888", fontstyle="italic")
    fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
             ha="left", va="bottom", fontsize=7.5, color="#555", fontstyle="italic")
    fig.text(0.99, 0.01, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888888",
             alpha=0.8, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    out = f"game_script_{SEASON}.png"
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


# ── MODO EQUIPO ────────────────────────────────────────────────────────────────
else:
    tm5   = plays[plays["posteam"] == team]
    tm4th = fourth[fourth["posteam"] == team]
    if tm5.empty:
        raise SystemExit(f"No hay jugadas para {team}.")

    # ── Estadísticas por bin ───────────────────────────────────────────────
    tm_stats = (tm5.groupby("bin5")
                .agg(epa=("epa", "mean"), pass_pct=("is_pass", "mean"), n=("epa", "count"))
                .reindex(BIN5_KEYS))
    lg_stats = (plays.groupby("bin5")
                .agg(lg_epa=("epa", "mean"), lg_pass_pct=("is_pass", "mean"))
                .reindex(BIN5_KEYS))

    # Enmascarar bins con pocas jugadas
    tm_stats.loc[tm_stats["n"] < MIN_PLAYS, ["epa", "pass_pct"]] = np.nan

    # PROE = pass_pct del equipo - lg_pass_pct (en el mismo bin)
    proe = (tm_stats["pass_pct"] - lg_stats["lg_pass_pct"]) * 100

    # 3er down por bin5
    third = tm5[tm5["down"] == 3].copy()
    for col in ["third_down_converted"]:
        if col in third.columns:
            third[col] = pd.to_numeric(third[col], errors="coerce")

    if "third_down_converted" in third.columns:
        td_stats = (third.groupby("bin5")
                    .agg(epa=("epa","mean"), conv=("third_down_converted","mean"), n=("epa","count"))
                    .reindex(BIN5_KEYS))
        lg_td = (plays[plays["down"]==3].groupby("bin5")
                 .agg(lg_conv=("third_down_converted","mean"))
                 .reindex(BIN5_KEYS))
    else:
        td_stats = (third.groupby("bin5")
                    .agg(epa=("epa","mean"), n=("epa","count"))
                    .reindex(BIN5_KEYS))
        td_stats["conv"] = np.nan
        lg_td = pd.DataFrame({"lg_conv": np.nan}, index=BIN5_KEYS)

    td_stats.loc[td_stats["n"] < 10, ["epa", "conv"]] = np.nan

    # 4th down decisiones por bin3
    if not tm4th.empty:
        d4_tm = (tm4th.groupby(["bin3", "decision"])["decision"]
                 .count().unstack(fill_value=0)
                 .reindex(BIN3_KEYS, fill_value=0))
        for col in ["go", "fg", "punt"]:
            if col not in d4_tm.columns:
                d4_tm[col] = 0
        d4_tm["total"] = d4_tm[["go","fg","punt"]].sum(axis=1)
        for col in ["go","fg","punt"]:
            d4_tm[f"{col}_pct"] = (d4_tm[col] / d4_tm["total"].replace(0, np.nan)) * 100
        d4_lg = (fourth.groupby(["bin3","decision"])["decision"]
                 .count().unstack(fill_value=0)
                 .reindex(BIN3_KEYS, fill_value=0))
        for col in ["go","fg","punt"]:
            if col not in d4_lg.columns:
                d4_lg[col] = 0
        d4_lg["total"] = d4_lg[["go","fg","punt"]].sum(axis=1)
        d4_lg["go_pct_lg"] = (d4_lg["go"] / d4_lg["total"].replace(0, np.nan)) * 100
    else:
        d4_tm = None

    # ── Figura 2×2 ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.07, right=0.97,
        top=0.86, bottom=0.12,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    x5 = np.arange(len(BIN5_KEYS))
    x3 = np.arange(len(BIN3_KEYS))

    # ── Panel 1: EPA por bin ──────────────────────────────────────────────
    style_ax(ax1)
    epa_vals = tm_stats["epa"].dropna()
    v_abs    = max(abs(epa_vals.min()), abs(epa_vals.max()), 0.05) if len(epa_vals) else 0.3
    norm_e   = Normalize(vmin=-v_abs, vmax=v_abs)

    for i, key in enumerate(BIN5_KEYS):
        val = tm_stats.loc[key, "epa"]
        n   = int(tm_stats.loc[key, "n"]) if not pd.isna(tm_stats.loc[key, "n"]) else 0
        if pd.isna(val):
            ax1.bar(i, 0.002, color=GRID, width=0.6, alpha=0.4, zorder=2)
            ax1.text(i, v_abs * 0.08, "—", ha="center", va="bottom", color="#555", fontsize=9)
        else:
            c = RYG(norm_e(val))
            ax1.bar(i, val, color=c, width=0.6, zorder=2, edgecolor=BG, linewidth=0.5)
            sign = "+" if val >= 0 else ""
            bar_label(ax1, i, val, f"{sign}{val:.3f}", v_abs, val >= 0)
            if abs(val) > v_abs * 0.2:
                lum = 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
                ax1.text(i, val * 0.45, f"n={n}", ha="center", va="center",
                         color="#0a0e13" if lum > 0.5 else FG, fontsize=6.5, zorder=3)

    ax1.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    # Liga average line
    for i, key in enumerate(BIN5_KEYS):
        lg = lg_stats.loc[key, "lg_epa"]
        if not pd.isna(lg):
            ax1.plot([i - 0.34, i + 0.34], [lg, lg],
                     color="#ffd166", linewidth=1.8, zorder=4)
    # Margen extra para que las etiquetas encima de barras no se corten
    y_lo, y_hi = ax1.get_ylim()
    ax1.set_ylim(y_lo - abs(y_lo) * 0.18, y_hi + abs(y_hi) * 0.18)
    ax1.set_xticks(x5)
    ax1.set_xticklabels(BIN5_XLBLS, color=FG, fontsize=8)
    ax1.set_ylabel("EPA / play", color=FG, fontsize=9)
    ax1.set_title("EPA ofensivo por contexto", color=FG, fontsize=10.5,
                  fontweight="bold", pad=8)
    ax1.legend(
        handles=[Line2D([0],[0], color="#ffd166", linewidth=2)],
        labels=["Media NFL"],
        facecolor=CARD, edgecolor=GRID, labelcolor=FG, fontsize=7.5, loc="best",
    )

    # ── Panel 2: Pass rate vs liga (PROE) ────────────────────────────────
    style_ax(ax2)
    proe_abs = max(abs(proe.dropna().min()), abs(proe.dropna().max()), 1.0) \
               if proe.notna().any() else 10
    norm_p   = Normalize(vmin=-proe_abs, vmax=proe_abs)

    for i, key in enumerate(BIN5_KEYS):
        val    = proe.get(key, np.nan)
        tm_raw = tm_stats.loc[key, "pass_pct"]
        lg_raw = lg_stats.loc[key, "lg_pass_pct"]
        if pd.isna(val) or pd.isna(tm_raw):
            ax2.bar(i, 0.1, color=GRID, width=0.6, alpha=0.4, zorder=2)
            ax2.text(i, 0.5, "—", ha="center", va="bottom", color="#555", fontsize=9)
        else:
            c = RYG(norm_p(val))
            ax2.bar(i, val, color=c, width=0.6, zorder=2, edgecolor=BG, linewidth=0.5)
            sign = "+" if val >= 0 else ""
            bar_label(ax2, i, val, f"{sign}{val:.1f}pp", proe_abs, val >= 0)
            # Valor absoluto del equipo dentro de la barra
            if abs(val) > proe_abs * 0.2:
                lum = 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
                ax2.text(i, val * 0.45, f"{tm_raw*100:.0f}%",
                         ha="center", va="center",
                         color="#0a0e13" if lum > 0.5 else FG, fontsize=6.5, zorder=3)

    ax2.axhline(0, color="#ffd166", linewidth=1.2, linestyle="--", alpha=0.7, zorder=1)
    ax2.set_xticks(x5)
    ax2.set_xticklabels(BIN5_XLBLS, color=FG, fontsize=8)
    ax2.set_ylabel("pp vs media NFL", color=FG, fontsize=9)
    ax2.set_title("Pass rate vs liga por contexto", color=FG, fontsize=10.5,
                  fontweight="bold", pad=8)
    # Margen extra para etiquetas
    y_lo2, y_hi2 = ax2.get_ylim()
    ax2.set_ylim(y_lo2 - abs(y_lo2) * 0.18, y_hi2 + abs(y_hi2) * 0.18)
    ax2.text(0.99, 0.99, "▲ más pase de lo esperado",
             transform=ax2.transAxes, ha="right", va="top",
             color="#06d6a0", fontsize=7, alpha=0.75)
    ax2.text(0.99, 0.01, "▼ más carrera de lo esperado",
             transform=ax2.transAxes, ha="right", va="bottom",
             color="#d84a4a", fontsize=7, alpha=0.75)

    # ── Panel 3: 3er down ────────────────────────────────────────────────
    style_ax(ax3)
    has_conv = td_stats["conv"].notna().any()

    for i, key in enumerate(BIN5_KEYS):
        val_epa  = td_stats.loc[key, "epa"]  if key in td_stats.index else np.nan
        val_conv = td_stats.loc[key, "conv"] if has_conv and key in td_stats.index else np.nan
        n        = int(td_stats.loc[key, "n"]) if key in td_stats.index and not pd.isna(td_stats.loc[key, "n"]) else 0
        lg_conv  = lg_td.loc[key, "lg_conv"]  if key in lg_td.index else np.nan

        if has_conv and not pd.isna(val_conv):
            c = RYG(Normalize(vmin=0.3, vmax=0.7)(val_conv))
            ax3.bar(i, val_conv * 100, color=c, width=0.6, zorder=2,
                    edgecolor=BG, linewidth=0.5)
            ax3.text(i, val_conv * 100 + 1.5, f"{val_conv*100:.0f}%",
                     ha="center", va="bottom", color=FG, fontsize=8.5, fontweight="bold")
            if not pd.isna(val_epa):
                sign = "+" if val_epa >= 0 else ""
                ax3.text(i, val_conv * 50, f"EPA {sign}{val_epa:.2f}",
                         ha="center", va="center", fontsize=6.5, color="#0a0e13", zorder=3)
            # Liga conversion line
            if not pd.isna(lg_conv):
                ax3.plot([i - 0.34, i + 0.34], [lg_conv * 100, lg_conv * 100],
                         color="#ffd166", linewidth=1.8, zorder=4)
        elif not pd.isna(val_epa):
            c = RYG(Normalize(vmin=-0.3, vmax=0.3)(val_epa))
            ax3.bar(i, abs(val_epa) * 100, color=c, width=0.6, zorder=2,
                    alpha=0.7, edgecolor=BG, linewidth=0.5,
                    bottom=0 if val_epa >= 0 else -abs(val_epa)*100)
        else:
            ax3.bar(i, 0.5, color=GRID, width=0.6, alpha=0.4, zorder=2)
            ax3.text(i, 1, "—", ha="center", va="bottom", color="#555", fontsize=9)

    ax3.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    ax3.set_xticks(x5)
    ax3.set_xticklabels(BIN5_XLBLS, color=FG, fontsize=8)
    ax3.set_ylabel("Tasa de conversión (%)" if has_conv else "EPA / play", color=FG, fontsize=9)
    ax3.set_title("3er down por contexto", color=FG, fontsize=10.5,
                  fontweight="bold", pad=8)
    if has_conv:
        ax3.set_ylim(0, 100)
        ax3.legend(
            handles=[Line2D([0],[0], color="#ffd166", linewidth=2)],
            labels=["Media NFL"], facecolor=CARD, edgecolor=GRID,
            labelcolor=FG, fontsize=7.5, loc="best",
        )

    # ── Panel 4: 4th down decisiones ─────────────────────────────────────
    style_ax(ax4)
    if d4_tm is not None and not d4_tm.empty:
        for i, key in enumerate(BIN3_KEYS):
            if key not in d4_tm.index:
                continue
            row    = d4_tm.loc[key]
            total  = row.get("total", 0)
            if total == 0:
                continue
            go_p   = row.get("go_pct",   0)
            fg_p   = row.get("fg_pct",   0)
            punt_p = row.get("punt_pct", 0)
            # Stacked bars
            ax4.bar(i, go_p,   width=0.6, color=COL_GO,   zorder=2, bottom=0,
                    edgecolor=BG, linewidth=0.4)
            ax4.bar(i, fg_p,   width=0.6, color=COL_FG,   zorder=2, bottom=go_p,
                    edgecolor=BG, linewidth=0.4)
            ax4.bar(i, punt_p, width=0.6, color=COL_PUNT, zorder=2, bottom=go_p + fg_p,
                    edgecolor=BG, linewidth=0.4)
            # Go-for-it % label
            if go_p >= 8:
                ax4.text(i, go_p / 2, f"{go_p:.0f}%",
                         ha="center", va="center", color="#0a0e13", fontsize=8, fontweight="bold")
            # Liga go-for-it line
            if key in d4_lg.index:
                lg_go = d4_lg.loc[key, "go_pct_lg"]
                if not pd.isna(lg_go):
                    ax4.plot([i - 0.34, i + 0.34], [lg_go, lg_go],
                             color="#ffd166", linewidth=1.8, zorder=4)
            # n= dentro de la barra de punt (evita solapamiento con leyenda)
            if punt_p >= 12:
                ax4.text(i, go_p + fg_p + punt_p * 0.5, f"n={int(total)}",
                         ha="center", va="center", color="#0a0e13", fontsize=6.5, zorder=3)
            else:
                ax4.text(i, min(go_p + fg_p + punt_p + 2, 97), f"n={int(total)}",
                         ha="center", va="bottom", color="#888", fontsize=6.5, zorder=3)

        ax4.set_xticks(x3)
        ax4.set_xticklabels(BIN3_KEYS, color=FG, fontsize=9)
        ax4.set_ylabel("% de decisiones", color=FG, fontsize=9)
        ax4.set_ylim(0, 110)
        ax4.set_title("4th down — decisiones por contexto", color=FG, fontsize=10.5,
                      fontweight="bold", pad=8)
        ax4.legend(
            handles=[
                Patch(color=COL_GO,   label="Go for it"),
                Patch(color=COL_FG,   label="Field Goal"),
                Patch(color=COL_PUNT, label="Punt"),
                Line2D([0],[0], color="#ffd166", linewidth=2, label="Go% media NFL"),
            ],
            facecolor=CARD, edgecolor=GRID, labelcolor=FG, fontsize=7.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
        )
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "Sin datos\nde 4th down", ha="center", va="center",
                 color="#555", fontsize=11, transform=ax4.transAxes)

    # ── Cabecera ──────────────────────────────────────────────────────────
    logo = load_logo(team, base_zoom=0.062)
    if logo:
        fig.add_artist(AnnotationBbox(logo, (0.06, 0.945), xycoords="figure fraction",
                                      frameon=False, zorder=5))
    fig.text(0.5, 0.965, f"{team}  ·  Game Script  ·  NFL {SEASON}",
             ha="center", va="top", fontsize=15, fontweight="bold", color=FG)
    fig.text(0.5, 0.928,
             f"EPA · Pass rate vs liga · 3er down · 4th down  —  según contexto de partido  "
             f"·  Barra amarilla = media NFL  ·  Mín. {MIN_PLAYS} jugadas",
             ha="center", va="top", fontsize=8, color="#888", fontstyle="italic")
    fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
             ha="left", va="bottom", fontsize=7.5, color="#555", fontstyle="italic")
    fig.text(0.99, 0.01, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888888",
             alpha=0.8, fontstyle="italic")

    out = f"game_script_{team}_{SEASON}.png"
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")
