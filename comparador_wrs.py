"""
comparador_wrs.py
Radar chart comparando dos WRs en 6 dimensiones.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, cargar_ngs, salida, season_cli, sello
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli()   # None = auto-detectar última temporada
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
DPI    = 170
LOGOS_DIR    = "logos"
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

MIN_TARGETS = 50   # minimum targets for a WR to be included in normalization

WR1_COLOR = "#06d6a0"
WR2_COLOR = "#ffd166"

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


def find_wr(all_wrs, query, df, receiver_col):
    """Return the name of the most-targeted WR matching query (case-insensitive)."""
    matches = [n for n in all_wrs if query.lower() in n.lower()]
    if not matches:
        raise SystemExit(f"WR '{query}' no encontrado. "
                         f"Prueba con otra parte del nombre.")
    best = max(matches, key=lambda n: df[df[receiver_col] == n].shape[0])
    if len(matches) > 1:
        print(f"  [!] '{query}' coincide con: {', '.join(matches)} >> seleccionado: {best}")
    return best


def compute_wr_metrics(wr_df):
    """
    Compute raw metric values for a WR given their targeted plays DataFrame.
    Returns a dict with keys matching METRIC_KEYS.
    """
    has_yardline = "yardline_100" in wr_df.columns and wr_df["yardline_100"].notna().any()
    has_down     = "down" in wr_df.columns and wr_df["down"].notna().any()
    has_yac      = "yards_after_catch" in wr_df.columns and wr_df["yards_after_catch"].notna().any()
    has_air      = "air_yards" in wr_df.columns and wr_df["air_yards"].notna().any()

    epa = wr_df["epa"]

    # 1. EPA/objetivo — mean EPA across all targeted plays
    epa_overall = epa.mean()

    # 2. Tasa de recepción — completions / targets
    catch_rate = wr_df["complete_pass"].mean() if "complete_pass" in wr_df.columns else float("nan")

    # 3. YAC/recepción — yards after catch per reception (completions only)
    if has_yac:
        completions = wr_df[wr_df["complete_pass"] == 1]
        yac = completions["yards_after_catch"].mean() if len(completions) >= 5 else float("nan")
    else:
        yac = float("nan")

    # 4. aDOT — average depth of target (air yards per target)
    adot = wr_df["air_yards"].mean() if has_air else float("nan")

    # 5. EPA zona roja (yardline_100 <= 20)
    if has_yardline:
        rz = wr_df[wr_df["yardline_100"] <= 20]["epa"]
        epa_rz = rz.mean() if len(rz) >= 5 else float("nan")
    else:
        epa_rz = float("nan")

    # 6. EPA 3er down
    if has_down:
        third = wr_df[wr_df["down"] == 3]["epa"]
        epa_3rd = third.mean() if len(third) >= 5 else float("nan")
    else:
        epa_3rd = float("nan")

    return {
        "EPA/objetivo":       epa_overall,
        "Separación":     SEPARACION.get(name, float("nan")),
        "YAC/recepción":      yac,
        "aDOT":               adot,
        "EPA zona roja":      epa_rz,
        "EPA 3er down":       epa_3rd,
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
wr1_input = input("WR 1 (apellido o nombre parcial, p.ej. Jefferson): ").strip()
wr2_input = input("WR 2 (apellido o nombre parcial, p.ej. Hill): ").strip()

# ── DATA ───────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)

# Separacion media al recibir (Next Gen Stats). Sustituye a "Tasa recepción",
# que correlaciona -0.747 con el aDOT (medido sobre 2025): medía la profundidad
# del objetivo, no la capacidad del receptor para desmarcarse.
SEPARACION = {}
try:
    _ngs, _ = cargar_ngs("receiving", SEASON)
    _ngs = _ngs[_ngs["week"] == 0]          # week 0 = agregado de temporada
    SEPARACION = dict(zip(_ngs["player_short_name"], _ngs["avg_separation"]))
    print(f"NGS: separacion de {len(SEPARACION)} receptores")
except Exception as e:
    print(f"  Aviso: sin datos de separacion NGS ({e}) — ese eje saldra N/D")
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

to_num(df, ["epa", "week", "down", "yardline_100", "air_yards",
            "yards_after_catch", "complete_pass", "pass_attempt"])

receiver_col = pick_col(df, "receiver_player_name", "receiver")

if receiver_col is None:
    raise SystemExit("No se encontró columna de nombre de receptor.")

# Filter to targeted plays with EPA and receiver name present
target_df = df[
    (df["pass_attempt"] == 1) &
    df["epa"].notna() &
    df[receiver_col].notna()
].copy()

print(f"Objetivos con EPA y receptor: {len(target_df):,}")

# ── FIND WRs ───────────────────────────────────────────────────────────────────
all_wrs  = target_df[receiver_col].dropna().unique()
wr1_name = find_wr(all_wrs, wr1_input, target_df, receiver_col)
wr2_name = find_wr(all_wrs, wr2_input, target_df, receiver_col)
print(f"Comparando: {wr1_name} vs {wr2_name}")

def _equipo(nombre):
    m = target_df.loc[target_df[receiver_col] == nombre, "posteam"].mode()
    return m.iloc[0] if len(m) else ""

team1 = _equipo(wr1_name)
team2 = _equipo(wr2_name)

# ── COMPUTE METRICS FOR ALL WRs (for normalization) ────────────────────────────
METRIC_KEYS = [
    "EPA/objetivo",
    "Separación",
    "YAC/recepción",
    "aDOT",
    "EPA zona roja",
    "EPA 3er down",
]

wr_counts = target_df.groupby(receiver_col)["epa"].count()
qualified_wrs = wr_counts[wr_counts >= MIN_TARGETS].index.tolist()

# Ensure our two WRs are included even if below threshold
for name in [wr1_name, wr2_name]:
    if name not in qualified_wrs:
        qualified_wrs.append(name)

print(f"WRs cualificados para normalización: {len(qualified_wrs)}")

all_raw = {}
for name in qualified_wrs:
    wr_sub = target_df[target_df[receiver_col] == name].copy()
    all_raw[name] = compute_wr_metrics(wr_sub)

# Build matrix for normalization
norm_df = pd.DataFrame(all_raw).T   # rows = WRs, cols = metrics

# Normalize each column 0-1 across all qualified WRs
norm_scaled = pd.DataFrame(index=norm_df.index)
for col in METRIC_KEYS:
    col_vals = pd.to_numeric(norm_df[col], errors="coerce")
    norm_scaled[col] = safe_norm_series(col_vals)

# Fill NaN with 0.5 (neutral)
norm_scaled = norm_scaled.fillna(0.5)

wr1_raw  = all_raw[wr1_name]
wr2_raw  = all_raw[wr2_name]
wr1_norm = norm_scaled.loc[wr1_name]
wr2_norm = norm_scaled.loc[wr2_name]

# ── CONSOLE TABLE ──────────────────────────────────────────────────────────────
print()
header = f"{'Metrica':<22} {wr1_name:<18} {wr2_name:<18}"
print(header)
print("-" * len(header))
for metric in METRIC_KEYS:
    v1 = wr1_raw[metric]
    v2 = wr2_raw[metric]
    s1 = f"{v1:+.3f}" if not pd.isna(v1) else "  N/D "
    s2 = f"{v2:+.3f}" if not pd.isna(v2) else "  N/D "
    print(f"{metric:<22} {s1:<18} {s2:<18}")
print()

# ── RADAR CHART ───────────────────────────────────────────────────────────────
N = len(METRIC_KEYS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

def make_radar_values(wr_norm_row):
    vals = [float(wr_norm_row[m]) for m in METRIC_KEYS]
    vals += vals[:1]
    return vals

v1_radar = make_radar_values(wr1_norm)
v2_radar = make_radar_values(wr2_norm)

fig = plt.figure(figsize=(8, 8), facecolor=BG)
fig.patch.set_facecolor(BG)

ax = fig.add_subplot(111, polar=True)
ax.set_facecolor("#151924")

# Grid and axis styling
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_rlim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["", "", "", ""], color=FG, fontsize=7)
ax.yaxis.grid(color=GRID, linewidth=0.8, alpha=0.5, linestyle="--")
ax.spines["polar"].set_color(GRID)

ax.set_xticks(angles[:-1])

# Axis labels with actual values
xticklabels = []
for i, metric in enumerate(METRIC_KEYS):
    v1_actual = wr1_raw[metric]
    v2_actual = wr2_raw[metric]
    s1 = f"{v1_actual:+.2f}" if not pd.isna(v1_actual) else "N/D"
    s2 = f"{v2_actual:+.2f}" if not pd.isna(v2_actual) else "N/D"
    label = f"{metric}\n{wr1_name.split('.')[-1].strip()}: {s1}\n{wr2_name.split('.')[-1].strip()}: {s2}"
    xticklabels.append(label)

ax.set_xticklabels(xticklabels, color=FG, fontsize=7.5, ha="center")

# Draw radar lines
ax.plot(angles, v1_radar, color=WR1_COLOR, linewidth=2.2, zorder=4, label=f"{wr1_name} ({team1})")
ax.fill(angles, v1_radar, color=WR1_COLOR, alpha=0.20, zorder=3)

ax.plot(angles, v2_radar, color=WR2_COLOR, linewidth=2.2, zorder=4, label=f"{wr2_name} ({team2})")
ax.fill(angles, v2_radar, color=WR2_COLOR, alpha=0.20, zorder=3)

# Draw reference ring at 0.5
ref_ring = [0.5] * (N + 1)
ax.plot(angles, ref_ring, color=GRID, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)

# Mark data points
ax.scatter(angles[:-1], v1_radar[:-1], color=WR1_COLOR, s=30, zorder=5)
ax.scatter(angles[:-1], v2_radar[:-1], color=WR2_COLOR, s=30, zorder=5)

# Legend
legend = ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.15),
    framealpha=0.25,
    facecolor="#151924",
    edgecolor=GRID,
    fontsize=9,
    labelcolor=FG,
)

# Tick label padding
for label in ax.get_xticklabels():
    label.set_color(FG)

# ── TITLES ─────────────────────────────────────────────────────────────────────
safe_wr1 = wr1_name.split(".")[-1].strip().replace(" ", "_")
safe_wr2 = wr2_name.split(".")[-1].strip().replace(" ", "_")

fig.text(0.5, 0.97, f"{wr1_name} vs {wr2_name}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         f"Comparación radar — 6 dimensiones | Normalizadas entre WRs con ≥{MIN_TARGETS} objetivos",
         ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  {sello(SEASON)}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = salida(f"comparador_{safe_wr1}_{safe_wr2}_{SEASON}.png", SEASON)
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
