"""
comparador_qbs.py
Radar chart comparando dos QBs en 6 dimensiones.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = 2025
URL    = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
DPI    = 200
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

MIN_PASS_PLAYS = 100   # minimum pass plays for a QB to be included in normalization

QB1_COLOR = "#06d6a0"
QB2_COLOR = "#ffd166"

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


def safe_norm_series(series):
    """Normalize a pd.Series 0-1. Returns series."""
    mn = series.min()
    mx = series.max()
    rng = mx - mn
    if rng == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / rng


def find_qb(all_qbs, query, df, passer_nm_col):
    """Return the name of the most-played QB matching query (case-insensitive)."""
    matches = [n for n in all_qbs if query.lower() in n.lower()]
    if not matches:
        raise SystemExit(f"QB '{query}' no encontrado. "
                         f"Prueba con otra parte del nombre.")
    # pick the one with most plays
    return max(matches, key=lambda n: df[df[passer_nm_col] == n].shape[0])


def compute_qb_metrics(qb_df, cols):
    """
    Compute raw metric values for a QB given their pass plays DataFrame.
    Returns a dict with keys matching METRIC_KEYS.
    """
    has_yardline = "yardline_100" in qb_df.columns and qb_df["yardline_100"].notna().any()
    has_down     = "down" in qb_df.columns and qb_df["down"].notna().any()
    has_cpoe     = "cpoe" in qb_df.columns and qb_df["cpoe"].notna().sum() > 10

    epa = qb_df["epa"]

    # 1. EPA global
    epa_overall = epa.mean()

    # 2. EPA Red Zone (yardline_100 <= 20)
    if has_yardline:
        rz = qb_df[qb_df["yardline_100"] <= 20]["epa"]
        epa_rz = rz.mean() if len(rz) > 0 else float("nan")
    else:
        epa_rz = float("nan")

    # 3. EPA 3rd down
    if has_down:
        third = qb_df[qb_df["down"] == 3]["epa"]
        epa_3rd = third.mean() if len(third) > 0 else float("nan")
    else:
        epa_3rd = float("nan")

    # 4. & 5. Pressure proxy
    pressured_mask = pd.Series(False, index=qb_df.index)
    if "qb_hit" in qb_df.columns:
        pressured_mask = pressured_mask | (pd.to_numeric(qb_df["qb_hit"], errors="coerce").fillna(0) == 1)
    if "sack" in qb_df.columns:
        pressured_mask = pressured_mask | (pd.to_numeric(qb_df["sack"], errors="coerce").fillna(0) == 1)

    press_plays = qb_df[pressured_mask]["epa"]
    clean_plays = qb_df[~pressured_mask]["epa"]
    epa_pressure = press_plays.mean() if len(press_plays) >= 5 else float("nan")
    epa_clean    = clean_plays.mean()  if len(clean_plays) >= 5 else float("nan")

    # 6. CPOE
    if has_cpoe:
        cpoe_val = qb_df["cpoe"].dropna().mean()
    else:
        cpoe_val = float("nan")

    return {
        "EPA global":         epa_overall,
        "EPA Red Zone":       epa_rz,
        "EPA 3a bajada":      epa_3rd,
        "EPA bajo presion":   epa_pressure,
        "EPA pocket limpio":  epa_clean,
        "CPOE":               cpoe_val,
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
qb1_input = input("QB 1 (apellido o nombre parcial, p.ej. Mahomes): ").strip()
qb2_input = input("QB 2 (apellido o nombre parcial, p.ej. Allen): ").strip()

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
print(f"Filas descargadas: {len(df):,}")

to_num(df, ["epa", "week", "down", "yardline_100", "qb_hit", "sack", "cpoe"])

passer_id_col = pick_col(df, "passer_player_id", "passer_id")
passer_nm_col = pick_col(df, "passer", "passer_player_name")

if passer_nm_col is None:
    raise SystemExit("No se encontro columna de nombre de passer.")

# Filter to pass plays with EPA and passer name present
pass_df = df[
    (df["play_type"] == "pass") &
    df["epa"].notna() &
    df[passer_nm_col].notna()
].copy()

print(f"Pases con EPA y passer: {len(pass_df):,}")

# ── FIND QBs ───────────────────────────────────────────────────────────────────
all_qbs   = pass_df[passer_nm_col].dropna().unique()
qb1_name  = find_qb(all_qbs, qb1_input, pass_df, passer_nm_col)
qb2_name  = find_qb(all_qbs, qb2_input, pass_df, passer_nm_col)
print(f"Comparando: {qb1_name} vs {qb2_name}")

# ── COMPUTE METRICS FOR ALL QBS (for normalization) ────────────────────────────
METRIC_KEYS = [
    "EPA global",
    "EPA Red Zone",
    "EPA 3a bajada",
    "EPA bajo presion",
    "EPA pocket limpio",
    "CPOE",
]

qb_counts = pass_df.groupby(passer_nm_col)["epa"].count()
qualified_qbs = qb_counts[qb_counts >= MIN_PASS_PLAYS].index.tolist()

# Ensure our two QBs are included even if below threshold
for name in [qb1_name, qb2_name]:
    if name not in qualified_qbs:
        qualified_qbs.append(name)

print(f"QBs qualificados para normalizacion: {len(qualified_qbs)}")

all_raw = {}
for name in qualified_qbs:
    qb_sub = pass_df[pass_df[passer_nm_col] == name].copy()
    all_raw[name] = compute_qb_metrics(qb_sub, METRIC_KEYS)

# Build matrix for normalization
norm_df = pd.DataFrame(all_raw).T   # rows = QBs, cols = metrics

# Normalize each column 0-1 across all qualified QBs
norm_scaled = pd.DataFrame(index=norm_df.index)
for col in METRIC_KEYS:
    col_vals = pd.to_numeric(norm_df[col], errors="coerce")
    norm_scaled[col] = safe_norm_series(col_vals)

# Fill NaN with 0.5 (neutral)
norm_scaled = norm_scaled.fillna(0.5)

qb1_raw  = all_raw[qb1_name]
qb2_raw  = all_raw[qb2_name]
qb1_norm = norm_scaled.loc[qb1_name]
qb2_norm = norm_scaled.loc[qb2_name]

# ── CONSOLE TABLE ──────────────────────────────────────────────────────────────
print()
header = f"{'Metrica':<22} {qb1_name:<18} {qb2_name:<18}"
print(header)
print("-" * len(header))
for metric in METRIC_KEYS:
    v1 = qb1_raw[metric]
    v2 = qb2_raw[metric]
    s1 = f"{v1:+.3f}" if not pd.isna(v1) else "  N/D "
    s2 = f"{v2:+.3f}" if not pd.isna(v2) else "  N/D "
    print(f"{metric:<22} {s1:<18} {s2:<18}")
print()

# ── RADAR CHART ───────────────────────────────────────────────────────────────
N = len(METRIC_KEYS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

def make_radar_values(qb_norm_row):
    vals = [float(qb_norm_row[m]) for m in METRIC_KEYS]
    vals += vals[:1]
    return vals

v1_radar = make_radar_values(qb1_norm)
v2_radar = make_radar_values(qb2_norm)

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
    v1_actual = qb1_raw[metric]
    v2_actual = qb2_raw[metric]
    s1 = f"{v1_actual:+.2f}" if not pd.isna(v1_actual) else "N/D"
    s2 = f"{v2_actual:+.2f}" if not pd.isna(v2_actual) else "N/D"
    label = f"{metric}\n{qb1_name.split('.')[-1].strip()}: {s1}\n{qb2_name.split('.')[-1].strip()}: {s2}"
    xticklabels.append(label)

ax.set_xticklabels(xticklabels, color=FG, fontsize=7.5, ha="center")

# Draw radar lines
ax.plot(angles, v1_radar, color=QB1_COLOR, linewidth=2.2, zorder=4, label=qb1_name)
ax.fill(angles, v1_radar, color=QB1_COLOR, alpha=0.20, zorder=3)

ax.plot(angles, v2_radar, color=QB2_COLOR, linewidth=2.2, zorder=4, label=qb2_name)
ax.fill(angles, v2_radar, color=QB2_COLOR, alpha=0.20, zorder=3)

# Draw reference ring at 0.5
ref_ring = [0.5] * (N + 1)
ax.plot(angles, ref_ring, color=GRID, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)

# Mark data points
ax.scatter(angles[:-1], v1_radar[:-1], color=QB1_COLOR, s=30, zorder=5)
ax.scatter(angles[:-1], v2_radar[:-1], color=QB2_COLOR, s=30, zorder=5)

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
safe_qb1 = qb1_name.split(".")[-1].strip().replace(" ", "_")
safe_qb2 = qb2_name.split(".")[-1].strip().replace(" ", "_")

fig.text(0.5, 0.97, f"{qb1_name} vs {qb2_name}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         f"Comparacion radar \u2014 6 dimensiones | Normalizadas entre QBs con \u2265{MIN_PASS_PLAYS} pases",
         ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  \u00b7  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = f"comparador_{safe_qb1}_{safe_qb2}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
