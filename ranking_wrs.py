"""
ranking_wrs.py
WRs — EPA/objetivo en Red Zone (X) vs 3er down (Y)
Tamaño del logo proporcional al nº de objetivos totales.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pbp_loader import cargar_pbp, cargar_stats, salida, season_cli
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── Config ────────────────────────────────────────────────────────────────────
SEASON       = season_cli()   # None = auto-detectar última temporada
LOGOS_DIR    = "logos"
BG           = "#0f1115"
FG           = "#EDEDED"
GRID         = "#2a2f3a"
DPI          = 170

MIN_RZ  = 10   # mínimo objetivos en zona roja
MIN_3RD = 15   # mínimo objetivos en 3er down

ZOOM_MIN = 0.021   # legible incluso con poco volumen
ZOOM_MAX = 0.042   # (escala pensada para la normalizacion por tinta real)

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_col(df, *cands):
    for c in cands:
        if c and c in df.columns:
            return c
    return None

def short_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    sufijos = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
    parts = [p for p in name.replace("-", " ").split()
             if p.lower() not in sufijos] or name.split()
    if len(parts) == 1:
        return parts[0][:14]
    return (parts[0][:1] + ". " + parts[-1])[:16]

def load_logo(team, zoom=0.030):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        # Recorta margenes transparentes: algunos archivos traen mucho aire
        # (NYJ: tinta 3768x1186 en lienzo 4096x4096) y sin recorte salen enanos
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        # Normaliza por el area de tinta real; los wordmarks apaisados pueden
        # ensancharse hasta 1.8x para compensar su poca altura
        z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * zoom:
            z = 900.0 * zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None

def separar_etiquetas(xs, lys, x_tol, y_tol):
    """Separa verticalmente etiquetas que caerian casi encima (los puntos
    no se mueven; solo el texto baja). lys = y iniciales de las etiquetas."""
    idx = sorted(range(len(xs)), key=lambda i: -lys[i])
    out = list(lys)
    for pos, i in enumerate(idx):
        for j in idx[:pos]:
            if abs(xs[i] - xs[j]) < x_tol and abs(out[i] - out[j]) < y_tol:
                out[i] = min(out[i], out[j] - y_tol)
    return out

def volume_zoom(n, n_min, n_max):
    if n_max == n_min:
        return (ZOOM_MIN + ZOOM_MAX) / 2
    t = (n - n_min) / (n_max - n_min)
    return ZOOM_MIN + t * (ZOOM_MAX - ZOOM_MIN)

# ── Datos ─────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
to_num(df, ["epa", "pass_attempt", "yardline_100", "down"])
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

df_stats, _ = cargar_stats(SEASON)
OUT = salida(f"scatter_WR_RZ_vs_3rd_{SEASON}.png", SEASON)
name_col_s = pick_col(df_stats, "player_name", "player_display_name")
disp_col_s = pick_col(df_stats, "player_display_name", "player_name")
pos_col_s  = pick_col(df_stats, "position", "pos")

wr_names    = set()
display_map = {}
if name_col_s and pos_col_s:
    for _, row in df_stats[df_stats[pos_col_s] == "WR"].iterrows():
        sn = row[name_col_s]
        if pd.isna(sn):
            continue
        wr_names.add(sn)
        display_map[sn] = row[disp_col_s] if disp_col_s else sn

receiver_col = pick_col(df, "receiver_player_name", "receiver")
if receiver_col is None:
    raise SystemExit("No se encontró columna de receptor en el PBP.")

target_df = df[
    (df["pass_attempt"] == 1) &
    df["epa"].notna() &
    df[receiver_col].notna()
].copy()

# ── Métricas por WR ───────────────────────────────────────────────────────────
sub = target_df[target_df[receiver_col].isin(wr_names)].copy()

rz  = sub[sub["yardline_100"] <= 20].groupby(receiver_col).agg(
    rz_epa=("epa", "mean"), rz_n=("epa", "count"))
d3  = sub[sub["down"] == 3].groupby(receiver_col).agg(
    d3_epa=("epa", "mean"), d3_n=("epa", "count"))
vol = sub.groupby(receiver_col).agg(total_n=("epa", "count"))

stats = rz.join(d3, how="inner").join(vol, how="inner")
stats = stats[(stats["rz_n"] >= MIN_RZ) & (stats["d3_n"] >= MIN_3RD)].copy()

teams = (
    sub.dropna(subset=[receiver_col, "posteam"])
    .groupby(receiver_col)["posteam"]
    .agg(lambda x: x.mode().iloc[0])
)
stats["team"]  = teams
stats["label"] = [short_name(display_map.get(n, n)) for n in stats.index]

print(f"\nWRs en el scatter: {len(stats)}")

avg_rz = stats["rz_epa"].mean()
avg_d3 = stats["d3_epa"].mean()
n_min  = stats["total_n"].min()
n_max  = stats["total_n"].max()

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9), dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.tick_params(colors=FG, labelsize=9)
plt.setp(ax.get_xticklabels(), color=FG)
plt.setp(ax.get_yticklabels(), color=FG)

x_vals = stats["rz_epa"].values
y_vals = stats["d3_epa"].values
x_pad  = (x_vals.max() - x_vals.min()) * 0.14
y_pad  = (y_vals.max() - y_vals.min()) * 0.14
ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

ax.axhline(avg_d3, color=GRID, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
ax.axvline(avg_rz, color=GRID, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
ax.grid(True, linestyle="--", alpha=0.12, color=GRID, zorder=0)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

x_lo, x_hi = ax.get_xlim()
y_lo, y_hi = ax.get_ylim()
xm = (x_hi - x_lo) * 0.03
ym = (y_hi - y_lo) * 0.03
q_kw = dict(fontsize=8.5, alpha=0.30, color=FG, fontstyle="italic")
ax.text(x_hi - xm, y_hi - ym, "Élite en ambas",            ha="right", va="top",    **q_kw)
ax.text(x_lo + xm, y_hi - ym, "Bueno en 3ro / Malo RZ",    ha="left",  va="top",    **q_kw)
ax.text(x_hi - xm, y_lo + ym, "Bueno en RZ / Malo 3ro",    ha="right", va="bottom", **q_kw)
ax.text(x_lo + xm, y_lo + ym, "Peor en ambas",             ha="left",  va="bottom", **q_kw)

y_range = y_hi - y_lo

# Offset bajo el punto proporcional al tamaño real del logo (los grandes
# necesitan más hueco) y anti-solape con tolerancias ajustadas al texto
zooms = [volume_zoom(n, n_min, n_max) for n in stats["total_n"]]
offs  = [y_range * (0.022 + 0.40 * z) for z in zooms]
lys = separar_etiquetas(
    stats["rz_epa"].tolist(),
    [y - off for y, off in zip(stats["d3_epa"], offs)],
    (x_hi - x_lo) * 0.060, y_range * 0.030)

for (wr_key, row), ly, off, zoom in zip(stats.iterrows(), lys, offs, zooms):
    x    = row["rz_epa"]
    y    = row["d3_epa"]
    team = row["team"]
    name = row["label"]

    logo = load_logo(str(team) if not pd.isna(team) else "", zoom=zoom)
    if logo:
        ab = AnnotationBbox(logo, (x, y), frameon=False, zorder=3,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.scatter(x, y, s=60 + 120 * (zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN),
                   color="#888888", zorder=3, alpha=0.8)

    # Si el anti-solape desplazó la etiqueta, línea guía logo → nombre
    if (y - off) - ly > y_range * 0.012:
        ax.plot([x, x], [y - off * 0.72, ly + y_range * 0.006],
                color=GRID, linewidth=0.7, alpha=0.9, zorder=2)

    ax.text(x, ly, name,
            ha="center", va="top", fontsize=7.5, color=FG, alpha=0.88, zorder=4,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=BG)])

ax.set_xlabel("EPA/objetivo en Red Zone", fontsize=11, color=FG, labelpad=7)
ax.set_ylabel("EPA/objetivo en 3er down", fontsize=11, color=FG, labelpad=7)
ax.set_title(f"WRs NFL {SEASON} — Red Zone vs 3er down",
             fontsize=15, pad=12, color=FG, fontweight="bold")


fig.text(0.5, 0.01,
         f"Fuente: nflverse-data  ·  mín. {MIN_RZ} obj. en RZ y {MIN_3RD} en 3er down  ·  Líneas = media de la muestra  ·  Tamaño del logo = nº de objetivos",
         ha="center", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato", fontsize=9, color="#888888",
         ha="right", va="bottom", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Guardado: {OUT}")
