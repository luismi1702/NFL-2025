"""
ataque_vs_personal_ofensivo.py
Heatmap: eficiencia ofensiva de cada equipo segun el personal
que usa en ataque (11, 12, 21, 10, 22...).
EPA generado: positivo = buen ataque.
NFL 2025
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = 2025
URL_PBP  = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
URL_PART = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{SEASON}.parquet"
BG       = "#0f1115"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 200
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

# Verde = EPA alto = buen ataque

MIN_SNAPS = 20   # snaps mínimos por celda para mostrar valor

# Paquetes principales a mostrar (en orden de frecuencia)
PKG_ORDER  = ["11", "12", "21", "10", "13", "22"]
PKG_LABELS = {
    "11": "11\n1RB·1TE·3WR",
    "12": "12\n1RB·2TE·2WR",
    "21": "21\n2RB·1TE·2WR",
    "10": "10\n1RB·0TE·4WR",
    "13": "13\n1RB·3TE·1WR",
    "22": "22\n2RB·2TE·1WR",
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.038):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        if team in HARD_PENALTY:
            zoom = base_zoom / HARD_PENALTY[team]
        else:
            div = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            zoom = base_zoom / div
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def parse_personnel(s):
    """
    Extrae RB y TE de cadenas tipo '1 RB, 1 TE, 3 WR'.
    Devuelve código de personal estilo '11', '12', '21', etc.
    """
    if pd.isna(s):
        return None
    s = str(s)
    rb = re.search(r"(\d+)\s*RB", s, re.IGNORECASE)
    te = re.search(r"(\d+)\s*TE", s, re.IGNORECASE)
    if rb and te:
        return f"{rb.group(1)}{te.group(1)}"
    return None


# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
pbp = pd.read_csv(URL_PBP, low_memory=False, compression="infer")
pbp["epa"]     = pd.to_numeric(pbp["epa"],     errors="coerce")
pbp["play_id"] = pd.to_numeric(pbp["play_id"], errors="coerce")
print(f"PBP filas: {len(pbp):,}")

print(f"Descargando participacion {SEASON}...")
part = pd.read_parquet(URL_PART, columns=["nflverse_game_id", "play_id", "offense_personnel"])
part = part.rename(columns={"nflverse_game_id": "game_id"})
part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")
print(f"Participacion filas: {len(part):,}")

# ── JOIN ───────────────────────────────────────────────────────────────────────
merged = pbp.merge(part, on=["game_id", "play_id"], how="left")

plays = merged[
    merged["play_type"].isin(["pass", "run"]) &
    merged["posteam"].notna() &
    merged["epa"].notna() &
    merged["offense_personnel"].notna()
].copy()

print(f"Jugadas con personal ofensivo: {len(plays):,}")

# ── CLASIFICAR PERSONAL OFENSIVO ───────────────────────────────────────────────
plays["pkg"] = plays["offense_personnel"].apply(parse_personnel)
plays = plays[plays["pkg"].isin(PKG_ORDER)].copy()

# ── AGREGAR POR POSTEAM × PAQUETE ─────────────────────────────────────────────
grp = (
    plays.groupby(["posteam", "pkg"])["epa"]
    .agg(epa_mean="mean", n_snaps="count")
    .reset_index()
)

# Pivot EPA y n_snaps
epa_piv = grp.pivot(index="posteam", columns="pkg", values="epa_mean")
n_piv   = grp.pivot(index="posteam", columns="pkg", values="n_snaps").fillna(0).astype(int)

# Asegurar todas las columnas
for col in PKG_ORDER:
    if col not in epa_piv.columns:
        epa_piv[col] = np.nan
    if col not in n_piv.columns:
        n_piv[col] = 0
epa_piv = epa_piv[PKG_ORDER]
n_piv   = n_piv[PKG_ORDER]

# Aplicar mínimo de snaps
epa_piv = epa_piv.where(n_piv >= MIN_SNAPS)

# Ordenar por EPA global ofensivo (mejor ataque arriba)
overall_epa = plays.groupby("posteam")["epa"].mean()
epa_piv["_sort"] = overall_epa
epa_piv = epa_piv.sort_values("_sort", ascending=False)
epa_piv = epa_piv.drop(columns="_sort")
n_piv   = n_piv.loc[epa_piv.index]

teams   = epa_piv.index.tolist()
n_teams = len(teams)

# ── CONSOLA ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  EPA generado por personal ofensivo | NFL {SEASON}  (+ = mejor ataque)")
print(f"{'='*70}")
header = f"{'Off':<5}" + "".join(f"  {'Pkg '+p:>10}" for p in PKG_ORDER)
print(header)
print("-" * len(header))
for tm in teams:
    row = epa_piv.loc[tm]
    vals = "".join(
        f"  {row[p]:+9.3f}" if not np.isnan(row[p]) else f"  {'N/D':>9}"
        for p in PKG_ORDER
    )
    print(f"{tm:<5}{vals}")
print()

# ── FIGURA ─────────────────────────────────────────────────────────────────────
n_cols  = len(PKG_ORDER)
cell_w  = 1.4
cell_h  = 0.52
logo_w  = 1.2
fig_w   = logo_w + n_cols * cell_w + 1.2
fig_h   = max(8, n_teams * cell_h + 2.5)

fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(-logo_w, n_cols)
ax.set_ylim(-1, n_teams + 0.8)

# ── COLORMAP ──────────────────────────────────────────────────────────────────
valid  = epa_piv.values[~np.isnan(epa_piv.values)]
v_abs  = max(abs(valid.min()), abs(valid.max()), 0.05) if len(valid) else 0.3
norm   = Normalize(vmin=-v_abs, vmax=v_abs)
# Verde = EPA alto (buen ataque), Rojo = EPA bajo (mal ataque)
cmap   = plt.cm.RdYlGn

# ── CELDAS ────────────────────────────────────────────────────────────────────
for row_i, team in enumerate(teams):
    y = n_teams - row_i - 1   # invertir para que mejor defensa quede arriba
    for col_j, pkg in enumerate(PKG_ORDER):
        val = epa_piv.loc[team, pkg]
        n   = n_piv.loc[team, pkg]
        x   = col_j

        bg_color = cmap(norm(val)) if not np.isnan(val) else "#1e2430"
        rect = plt.Rectangle((x, y), 1, 1, color=bg_color,
                              linewidth=0.4, edgecolor=BG, zorder=1)
        ax.add_patch(rect)

        if not np.isnan(val):
            sign      = "+" if val >= 0 else ""
            txt_color = "#0a0e13" if 0.3 < norm(val) < 0.7 else FG
            ax.text(x + 0.5, y + 0.58, f"{sign}{val:.3f}",
                    ha="center", va="center",
                    color=txt_color, fontsize=8, fontweight="bold", zorder=2)
            ax.text(x + 0.5, y + 0.25, f"n={n}",
                    ha="center", va="center",
                    color=txt_color, fontsize=6.5, zorder=2)
        else:
            ax.text(x + 0.5, y + 0.5, "—",
                    ha="center", va="center",
                    color="#444444", fontsize=10, zorder=2)

# ── LOGOS ──────────────────────────────────────────────────────────────────────
for row_i, team in enumerate(teams):
    y = n_teams - row_i - 1
    img = load_logo(team, base_zoom=0.036)
    if img is not None:
        ab = AnnotationBbox(img, (-logo_w / 2, y + 0.5),
                            frameon=False, zorder=3,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.text(-logo_w / 2, y + 0.5, team,
                ha="center", va="center",
                color=FG, fontsize=7.5, fontweight="bold")

# ── CABECERAS DE COLUMNA ──────────────────────────────────────────────────────
for col_j, pkg in enumerate(PKG_ORDER):
    ax.text(col_j + 0.5, n_teams + 0.35,
            PKG_LABELS[pkg],
            ha="center", va="center",
            color=FG, fontsize=8, fontweight="bold", linespacing=1.3)

# Línea separadora cabecera
ax.axhline(n_teams, color=GRID, linewidth=0.8, xmin=0, xmax=1, zorder=3)

# ── COLORBAR ─────────────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.70])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("EPA generado\n(+ = mejor ataque)", color=FG, fontsize=8)
cb.ax.yaxis.set_tick_params(color=FG, labelsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
cb.outline.set_edgecolor(GRID)

# ── TÍTULOS ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.99,
         f"Eficiencia ofensiva por personal | NFL {SEASON}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.975,
         "EPA generado segun el paquete ofensivo usado  |  Ordenado por EPA global (mejor ataque arriba)  |  — = menos de 20 snaps",
         ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
fig.text(0.01, 0.005, f"Fuente: nflverse-data + NGS participation  |  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.90, 0.005, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

outfile = f"ataque_vs_personal_ofensivo_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
