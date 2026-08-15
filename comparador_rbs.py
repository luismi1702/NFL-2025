"""
comparador_rbs.py
Radar chart comparando dos RBs en 6 dimensiones (todas del PBP).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, salida, season_cli, sello
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = season_cli()   # None = auto-detectar última temporada
BG       = "#0f1115"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 170
LOGOS_DIR    = "logos"
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

MIN_CARRIES = 50   # minimum carries for a RB to be included in normalization

RB1_COLOR = "#06d6a0"
RB2_COLOR = "#ffd166"

# ── HELPERS ────────────────────────────────────────────────────────────────────
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


def load_logo(team, base_zoom=0.055):
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
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        zoom = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * zoom > 900.0 * (base_zoom):
            zoom = 900.0 * (base_zoom) / w
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def safe_norm_series(series):
    """Normalize a pd.Series 0-1. Returns series."""
    mn = series.min()
    mx = series.max()
    rng = mx - mn
    if rng == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / rng


def find_rb(all_rbs, query, df, rusher_col):
    """Return the name of the most-used RB matching query (case-insensitive)."""
    matches = [n for n in all_rbs if query.lower() in n.lower()]
    if not matches:
        raise SystemExit(f"RB '{query}' no encontrado. "
                         f"Prueba con otra parte del nombre.")
    best = max(matches, key=lambda n: df[df[rusher_col] == n].shape[0])
    if len(matches) > 1:
        print(f"  [!] '{query}' coincide con: {', '.join(matches)} >> seleccionado: {best}")
    return best


def compute_rb_metrics(rush_sub, recv_sub):
    """
    Compute raw metric values for a RB.
    rush_sub: DataFrame of rush plays for this RB
    recv_sub: DataFrame of targeted pass plays for this RB
    """
    has_yardline = "yardline_100" in rush_sub.columns and rush_sub["yardline_100"].notna().any()

    # 1. EPA/acarreo
    epa_rush = rush_sub["epa"].mean()

    # 2. YPC
    ypc = rush_sub["yards_gained"].mean() if "yards_gained" in rush_sub.columns else float("nan")

    # 3. Explosividad — % de carreras con 10+ yards
    if "yards_gained" in rush_sub.columns and len(rush_sub) > 0:
        explosividad = (rush_sub["yards_gained"] >= 10).mean()
    else:
        explosividad = float("nan")

    # 4. EPA zona roja (rush plays inside the 20)
    if has_yardline:
        rz = rush_sub[rush_sub["yardline_100"] <= 20]["epa"]
        epa_rz = rz.mean() if len(rz) >= 5 else float("nan")
    else:
        epa_rz = float("nan")

    # 5. EPA como receptor
    epa_recv = recv_sub["epa"].mean() if len(recv_sub) >= 5 else float("nan")

    # 6. Tasa de éxito — % de carreras con EPA positivo
    tasa_exito = (rush_sub["epa"] > 0).mean() if len(rush_sub) > 0 else float("nan")

    return {
        "EPA/acarreo":        epa_rush,
        "YPC":                ypc,
        "Explosividad 10+":   explosividad,
        "EPA zona roja":      epa_rz,
        "EPA como receptor":  epa_recv,
        "Tasa de éxito":      tasa_exito,
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
rb1_input = input("RB 1 (apellido o nombre parcial, p.ej. McCaffrey): ").strip()
rb2_input = input("RB 2 (apellido o nombre parcial, p.ej. Henry): ").strip()

# ── DATA — PBP ─────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

to_num(df, ["epa", "week", "down", "yardline_100", "yards_gained", "pass_attempt"])

rusher_col   = pick_col(df, "rusher_player_name", "rusher")
receiver_col = pick_col(df, "receiver_player_name", "receiver")

if rusher_col is None:
    raise SystemExit("No se encontró columna de nombre de corredor.")

# Rush plays
rush_df = df[
    (df["play_type"] == "run") &
    df["epa"].notna() &
    df[rusher_col].notna()
].copy()
print(f"Carreras con EPA y corredor: {len(rush_df):,}")

# Receive plays (for EPA como receptor)
if receiver_col:
    recv_df = df[
        (df["pass_attempt"] == 1) &
        df["epa"].notna() &
        df[receiver_col].notna()
    ].copy()
else:
    recv_df = pd.DataFrame()


# ── FIND RBs ───────────────────────────────────────────────────────────────────
all_rbs  = rush_df[rusher_col].dropna().unique()
rb1_name = find_rb(all_rbs, rb1_input, rush_df, rusher_col)
rb2_name = find_rb(all_rbs, rb2_input, rush_df, rusher_col)
print(f"Comparando: {rb1_name} vs {rb2_name}")

def _equipo(nombre):
    m = rush_df.loc[rush_df[rusher_col] == nombre, "posteam"].mode()
    return m.iloc[0] if len(m) else ""

team1 = _equipo(rb1_name)
team2 = _equipo(rb2_name)

# ── COMPUTE METRICS FOR ALL RBs (for normalization) ────────────────────────────
METRIC_KEYS = [
    "EPA/acarreo",
    "YPC",
    "Explosividad 10+",
    "EPA zona roja",
    "EPA como receptor",
    "Tasa de éxito",
]

rb_counts     = rush_df.groupby(rusher_col)["epa"].count()
qualified_rbs = rb_counts[rb_counts >= MIN_CARRIES].index.tolist()

for name in [rb1_name, rb2_name]:
    if name not in qualified_rbs:
        qualified_rbs.append(name)

print(f"RBs cualificados para normalización: {len(qualified_rbs)}")

all_raw = {}
for name in qualified_rbs:
    rush_sub = rush_df[rush_df[rusher_col] == name].copy()
    recv_sub = recv_df[recv_df[receiver_col] == name].copy() if receiver_col and len(recv_df) > 0 else pd.DataFrame()
    all_raw[name] = compute_rb_metrics(rush_sub, recv_sub)

# Build matrix for normalization
norm_df = pd.DataFrame(all_raw).T

norm_scaled = pd.DataFrame(index=norm_df.index)
for col in METRIC_KEYS:
    col_vals = pd.to_numeric(norm_df[col], errors="coerce")
    norm_scaled[col] = safe_norm_series(col_vals)

norm_scaled = norm_scaled.fillna(0.5)

rb1_raw  = all_raw[rb1_name]
rb2_raw  = all_raw[rb2_name]
rb1_norm = norm_scaled.loc[rb1_name]
rb2_norm = norm_scaled.loc[rb2_name]

# ── CONSOLE TABLE ──────────────────────────────────────────────────────────────
print()
header = f"{'Metrica':<22} {rb1_name:<18} {rb2_name:<18}"
print(header)
print("-" * len(header))
for metric in METRIC_KEYS:
    v1 = rb1_raw[metric]
    v2 = rb2_raw[metric]
    s1 = f"{v1:+.3f}" if not pd.isna(v1) else "  N/D "
    s2 = f"{v2:+.3f}" if not pd.isna(v2) else "  N/D "
    print(f"{metric:<22} {s1:<18} {s2:<18}")
print()

# ── RADAR CHART ───────────────────────────────────────────────────────────────
N = len(METRIC_KEYS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def make_radar_values(rb_norm_row):
    vals = [float(rb_norm_row[m]) for m in METRIC_KEYS]
    vals += vals[:1]
    return vals

v1_radar = make_radar_values(rb1_norm)
v2_radar = make_radar_values(rb2_norm)

fig = plt.figure(figsize=(8, 8), facecolor=BG)
fig.patch.set_facecolor(BG)

ax = fig.add_subplot(111, polar=True)
ax.set_facecolor("#151924")

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_rlim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["", "", "", ""], color=FG, fontsize=7)
ax.yaxis.grid(color=GRID, linewidth=0.8, alpha=0.5, linestyle="--")
ax.spines["polar"].set_color(GRID)

ax.set_xticks(angles[:-1])

xticklabels = []
for i, metric in enumerate(METRIC_KEYS):
    v1_actual = rb1_raw[metric]
    v2_actual = rb2_raw[metric]
    s1 = f"{v1_actual:+.2f}" if not pd.isna(v1_actual) else "N/D"
    s2 = f"{v2_actual:+.2f}" if not pd.isna(v2_actual) else "N/D"
    label = f"{metric}\n{rb1_name.split('.')[-1].strip()}: {s1}\n{rb2_name.split('.')[-1].strip()}: {s2}"
    xticklabels.append(label)

ax.set_xticklabels(xticklabels, color=FG, fontsize=7.5, ha="center")

ax.plot(angles, v1_radar, color=RB1_COLOR, linewidth=2.2, zorder=4, label=f"{rb1_name} ({team1})")
ax.fill(angles, v1_radar, color=RB1_COLOR, alpha=0.20, zorder=3)

ax.plot(angles, v2_radar, color=RB2_COLOR, linewidth=2.2, zorder=4, label=f"{rb2_name} ({team2})")
ax.fill(angles, v2_radar, color=RB2_COLOR, alpha=0.20, zorder=3)

ref_ring = [0.5] * (N + 1)
ax.plot(angles, ref_ring, color=GRID, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)

ax.scatter(angles[:-1], v1_radar[:-1], color=RB1_COLOR, s=30, zorder=5)
ax.scatter(angles[:-1], v2_radar[:-1], color=RB2_COLOR, s=30, zorder=5)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.15),
    framealpha=0.25,
    facecolor="#151924",
    edgecolor=GRID,
    fontsize=9,
    labelcolor=FG,
)

for label in ax.get_xticklabels():
    label.set_color(FG)

# ── TITLES ─────────────────────────────────────────────────────────────────────
safe_rb1 = rb1_name.split(".")[-1].strip().replace(" ", "_")
safe_rb2 = rb2_name.split(".")[-1].strip().replace(" ", "_")

fig.text(0.5, 0.97, f"{rb1_name} vs {rb2_name}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         f"Comparación radar — 6 dimensiones | Normalizadas entre RBs con ≥{MIN_CARRIES} carreras",
         ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  {sello(SEASON)}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = salida(f"comparador_{safe_rb1}_{safe_rb2}_{SEASON}.png", SEASON)
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
