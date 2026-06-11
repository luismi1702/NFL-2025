"""
qb_overview.py
Scatter: EPA/play (X) vs CPOE (Y) — todos los QBs de la liga.
Tamaño del halo = volumen de intentos · Color = EPA bajo presión.
Logos de equipo en cada punto. Líneas de referencia en medias de liga.
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pbp_loader import cargar_pbp
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SEASON       = None  # None = auto-detectar última temporada
MIN_ATTEMPTS = 150   # intentos mínimos para aparecer
MIN_CPOE     =  50   # pases con CPOE válido mínimos

BG    = "#0f1115"
FG    = "#EDEDED"
GRID  = "#2a2f3a"
DPI   = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])


def load_logo(team, base_zoom=0.033):
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


def short_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {parts[-1]}"


# ── DATA ───────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

for col in ["epa", "cpoe"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

id_col   = next((c for c in ["passer_player_id", "passer_id"] if c in df.columns), None)
name_col = next((c for c in ["passer", "passer_player_name"] if c in df.columns), None)

if id_col is None:
    raise SystemExit("No se encontró columna de ID de passer.")

pass_df = df[
    (df["play_type"] == "pass") &
    df["epa"].notna() &
    df[id_col].notna()
].copy()

print(f"Pases con EPA: {len(pass_df):,}")

# ── AGGREGATE ─────────────────────────────────────────────────────────────────
# Detección de presión
press_col = "was_pressure" if "was_pressure" in pass_df.columns else None
if press_col:
    pass_df["pressured"] = pd.to_numeric(pass_df[press_col], errors="coerce").fillna(0).astype(bool)
else:
    pass_df["pressured"] = False
    for c in ["qb_hit", "sack"]:
        if c in pass_df.columns:
            pass_df["pressured"] |= pd.to_numeric(pass_df[c], errors="coerce").fillna(0).eq(1)

grp = pass_df.groupby(id_col).agg(
    epa_mean  = ("epa",  "mean"),
    cpoe_mean = ("cpoe", "mean"),
    n_att     = ("epa",  "count"),
    n_cpoe    = ("cpoe", lambda s: s.notna().sum()),
).reset_index()

press_grp = (pass_df[pass_df["pressured"]]
             .groupby(id_col)
             .agg(epa_press=("epa", "mean"), n_press=("epa", "count"))
             .reset_index())
grp = grp.merge(press_grp, on=id_col, how="left")

grp = grp[
    (grp["n_att"]  >= MIN_ATTEMPTS) &
    (grp["n_cpoe"] >= MIN_CPOE)
].copy()

if name_col:
    name_map = (pass_df.dropna(subset=[id_col, name_col])
                .groupby(id_col)[name_col]
                .agg(lambda s: s.value_counts().index[0]))
    grp["name"] = grp[id_col].map(name_map).fillna("")
else:
    grp["name"] = grp[id_col]

team_map = (pass_df.dropna(subset=[id_col, "posteam"])
            .groupby(id_col)["posteam"]
            .agg(lambda s: s.value_counts().index[0]))
grp["team"]  = grp[id_col].map(team_map).fillna("")
grp["label"] = grp["name"].apply(short_name)

print(f"\nQBs en el scatter: {len(grp)}")
print(grp[["label","epa_mean","cpoe_mean","epa_press","n_att"]]
      .sort_values("epa_mean", ascending=False).to_string(index=False))

# ── MEDIA DE LIGA ─────────────────────────────────────────────────────────────
lg_epa  = pass_df["epa"].mean()
lg_cpoe = pass_df["cpoe"].dropna().mean()

# ── FIGURA ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9), facecolor=BG)
ax.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID)
ax.tick_params(colors=FG, labelsize=9)
plt.setp(ax.get_xticklabels(), color=FG)
plt.setp(ax.get_yticklabels(), color=FG)

# Márgenes
x_vals = grp["epa_mean"].values
y_vals = grp["cpoe_mean"].values
x_pad  = (x_vals.max() - x_vals.min()) * 0.16
y_pad  = (y_vals.max() - y_vals.min()) * 0.16
ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

x_lo, x_hi = ax.get_xlim()
y_lo, y_hi = ax.get_ylim()

# Líneas de media de liga
ax.axvline(lg_epa,  color="#4a556a", linewidth=1.1, linestyle="--", alpha=0.8, zorder=1)
ax.axhline(lg_cpoe, color="#4a556a", linewidth=1.1, linestyle="--", alpha=0.8, zorder=1)
ax.text(lg_epa + (x_hi - x_lo) * 0.01, y_hi - (y_hi - y_lo) * 0.02,
        f"liga {lg_epa:+.3f}", color="#4a556a", fontsize=7, va="top")
ax.text(x_hi - (x_hi - x_lo) * 0.01, lg_cpoe + (y_hi - y_lo) * 0.01,
        f"liga {lg_cpoe:+.1f}%", color="#4a556a", fontsize=7, ha="right")

ax.grid(True, linestyle="--", alpha=0.08, color=GRID, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Etiquetas de cuadrante (en relación a medias de liga)
xm = (x_hi - x_lo) * 0.025
ym = (y_hi - y_lo) * 0.025
q_kw = dict(fontsize=8.5, alpha=0.25, color=FG, fontstyle="italic")
ax.text(x_hi - xm, y_hi - ym, "Élite",                       ha="right", va="top",    **q_kw)
ax.text(x_lo + xm, y_hi - ym, "Preciso\npero ineficiente",    ha="left",  va="top",    **q_kw)
ax.text(x_hi - xm, y_lo + ym, "Genera valor\nsin precisión",  ha="right", va="bottom", **q_kw)
ax.text(x_lo + xm, y_lo + ym, "Necesita\nmejorar",            ha="left",  va="bottom", **q_kw)

# ── COLOR + TAMAÑO ────────────────────────────────────────────────────────────
press_vals = grp["epa_press"].fillna(grp["epa_mean"])   # fallback si no hay datos de presión
norm_d = Normalize(vmin=press_vals.min(), vmax=press_vals.max())

att_vals = grp["n_att"].values.astype(float)
att_norm = (att_vals - att_vals.min()) / (att_vals.max() - att_vals.min() + 1e-9)
sizes    = 100 + att_norm * 520   # 100–620 px²

# Halo coloreado por EPA bajo presión detrás del logo
ax.scatter(
    grp["epa_mean"], grp["cpoe_mean"],
    s=sizes,
    c=press_vals,
    cmap=RYG, norm=norm_d,
    edgecolors="white", linewidths=0.4,
    alpha=0.30, zorder=2,
)

# Logos + nombres
y_range   = y_hi - y_lo
label_off = y_range * 0.036

for _, row in grp.iterrows():
    x, y = row["epa_mean"], row["cpoe_mean"]
    logo = load_logo(row["team"])
    if logo:
        ab = AnnotationBbox(logo, (x, y), frameon=False, zorder=4, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.scatter(x, y, s=80, color=RYG(norm_d(row["dakota"])), zorder=4)

    ax.text(x, y - label_off, row["label"],
            ha="center", va="top", fontsize=7.5, color=FG, alpha=0.90, zorder=5,
            path_effects=[pe.Stroke(linewidth=1.8, foreground=BG), pe.Normal()])

# ── COLORBAR ──────────────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=RYG, norm=norm_d)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.72)
cbar.set_label("EPA/play bajo presión", color=FG, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG, fontsize=7.5)
cbar.outline.set_edgecolor(GRID)

# ── EJES Y TÍTULOS ────────────────────────────────────────────────────────────
ax.set_xlabel("EPA / play",                         color=FG, fontsize=11, labelpad=7)
ax.set_ylabel("CPOE  (% completaciones sobre lo esperado)", color=FG, fontsize=11, labelpad=7)

fig.text(0.5, 0.97,
         f"QBs NFL {SEASON}  —  EPA/play vs CPOE",
         ha="center", va="top", fontsize=15, fontweight="bold", color=FG)
fig.text(0.5, 0.925,
         f"Halo = volumen de intentos  ·  Color = EPA bajo presión  ·  "
         f"Líneas punteadas = media de liga  ·  Mín. {MIN_ATTEMPTS} intentos",
         ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = f"qb_overview_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
