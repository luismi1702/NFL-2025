"""
comparador_tes.py
Radar chart comparando dos TEs en 6 dimensiones (5 del PBP + WOPR de stats_player).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, cargar_stats
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON    = None   # None = auto-detectar última temporada
BG        = "#0f1115"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
DPI       = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

MIN_TARGETS = 30   # minimum targets for a TE to be included in normalization

TE1_COLOR = "#06d6a0"
TE2_COLOR = "#ffd166"

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


def find_te(all_tes, query, df, receiver_col):
    """Return the name of the most-targeted TE matching query (case-insensitive)."""
    matches = [n for n in all_tes if query.lower() in n.lower()]
    if not matches:
        raise SystemExit(f"TE '{query}' no encontrado. "
                         f"Prueba con otra parte del nombre.")
    best = max(matches, key=lambda n: df[df[receiver_col] == n].shape[0])
    if len(matches) > 1:
        print(f"  [!] '{query}' coincide con: {', '.join(matches)} >> seleccionado: {best}")
    return best


def get_wopr(te_name_pbp, df_stats):
    """
    Look up WOPR for a TE from stats_player.
    player_name in stats_player uses the same short format as PBP (e.g. 'T.Kelce').
    Returns wopr value or nan.
    """
    if "wopr" not in df_stats.columns:
        return float("nan")
    nm_col = pick_col(df_stats, "player_name", "player_display_name")
    if nm_col is None:
        return float("nan")
    # Try exact match first
    match = df_stats[df_stats[nm_col] == te_name_pbp]
    if len(match) == 0:
        # Fallback: match by last name
        last = te_name_pbp.split(".")[-1].strip().lower()
        match = df_stats[df_stats[nm_col].str.lower().str.contains(last, na=False)]
    if len(match) == 0:
        return float("nan")
    return float(match.iloc[0]["wopr"])


def compute_te_metrics(te_df, wopr_val):
    """
    Compute raw metric values for a TE.
    te_df:    DataFrame of targeted pass plays for this TE (from PBP)
    wopr_val: float, WOPR from stats_player
    """
    has_yardline = "yardline_100" in te_df.columns and te_df["yardline_100"].notna().any()
    has_down     = "down" in te_df.columns and te_df["down"].notna().any()
    has_yac      = "yards_after_catch" in te_df.columns and te_df["yards_after_catch"].notna().any()

    # 1. EPA/objetivo
    epa_overall = te_df["epa"].mean()

    # 2. Tasa de recepción
    catch_rate = te_df["complete_pass"].mean() if "complete_pass" in te_df.columns else float("nan")

    # 3. YAC/recepción (completions only)
    if has_yac:
        completions = te_df[te_df["complete_pass"] == 1]
        yac = completions["yards_after_catch"].mean() if len(completions) >= 5 else float("nan")
    else:
        yac = float("nan")

    # 4. WOPR (from stats_player)
    wopr = float(wopr_val) if not pd.isna(wopr_val) else float("nan")

    # 5. EPA zona roja (yardline_100 <= 20)
    if has_yardline:
        rz = te_df[te_df["yardline_100"] <= 20]["epa"]
        epa_rz = rz.mean() if len(rz) >= 5 else float("nan")
    else:
        epa_rz = float("nan")

    # 6. EPA 3er down
    if has_down:
        third = te_df[te_df["down"] == 3]["epa"]
        epa_3rd = third.mean() if len(third) >= 5 else float("nan")
    else:
        epa_3rd = float("nan")

    return {
        "EPA/objetivo":    epa_overall,
        "Tasa recepción":  catch_rate,
        "YAC/recepción":   yac,
        "WOPR":            wopr,
        "EPA zona roja":   epa_rz,
        "EPA 3er down":    epa_3rd,
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
te1_input = input("TE 1 (apellido o nombre parcial, p.ej. Kelce): ").strip()
te2_input = input("TE 2 (apellido o nombre parcial, p.ej. Andrews): ").strip()

# ── DATA — PBP ─────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

to_num(df, ["epa", "week", "down", "yardline_100", "air_yards",
            "yards_after_catch", "complete_pass", "pass_attempt"])

receiver_col = pick_col(df, "receiver_player_name", "receiver")

if receiver_col is None:
    raise SystemExit("No se encontró columna de nombre de receptor.")

target_df = df[
    (df["pass_attempt"] == 1) &
    df["epa"].notna() &
    df[receiver_col].notna()
].copy()

print(f"Objetivos con EPA y receptor: {len(target_df):,}")

# ── DATA — STATS PLAYER ────────────────────────────────────────────────────────
try:
    df_stats, _ = cargar_stats(SEASON)
    to_num(df_stats, ["wopr", "targets"])
    # Keep only TEs with enough targets for WOPR normalization context
    te_stats = df_stats[
        (df_stats["position"] == "TE") &
        (df_stats["targets"] >= MIN_TARGETS)
    ].copy()
    print(f"TEs en stats_player con ≥{MIN_TARGETS} objetivos: {len(te_stats)}")
except Exception as e:
    print(f"Aviso: no se pudo cargar stats_player ({e}). WOPR = N/D.")
    df_stats = pd.DataFrame()
    te_stats  = pd.DataFrame()

# ── FIND TEs ───────────────────────────────────────────────────────────────────
all_tes  = target_df[receiver_col].dropna().unique()
te1_name = find_te(all_tes, te1_input, target_df, receiver_col)
te2_name = find_te(all_tes, te2_input, target_df, receiver_col)
print(f"Comparando: {te1_name} vs {te2_name}")

# ── COMPUTE METRICS FOR ALL TEs (for normalization) ────────────────────────────
METRIC_KEYS = [
    "EPA/objetivo",
    "Tasa recepción",
    "YAC/recepción",
    "WOPR",
    "EPA zona roja",
    "EPA 3er down",
]

te_counts     = target_df.groupby(receiver_col)["epa"].count()
qualified_tes = te_counts[te_counts >= MIN_TARGETS].index.tolist()

for name in [te1_name, te2_name]:
    if name not in qualified_tes:
        qualified_tes.append(name)

print(f"TEs cualificados para normalización: {len(qualified_tes)}")

all_raw = {}
for name in qualified_tes:
    te_sub   = target_df[target_df[receiver_col] == name].copy()
    wopr_val = get_wopr(name, df_stats) if len(df_stats) > 0 else float("nan")
    all_raw[name] = compute_te_metrics(te_sub, wopr_val)

# Build matrix for normalization
norm_df = pd.DataFrame(all_raw).T

norm_scaled = pd.DataFrame(index=norm_df.index)
for col in METRIC_KEYS:
    col_vals = pd.to_numeric(norm_df[col], errors="coerce")
    norm_scaled[col] = safe_norm_series(col_vals)

norm_scaled = norm_scaled.fillna(0.5)

te1_raw  = all_raw[te1_name]
te2_raw  = all_raw[te2_name]
te1_norm = norm_scaled.loc[te1_name]
te2_norm = norm_scaled.loc[te2_name]

# ── CONSOLE TABLE ──────────────────────────────────────────────────────────────
print()
header = f"{'Metrica':<22} {te1_name:<18} {te2_name:<18}"
print(header)
print("-" * len(header))
for metric in METRIC_KEYS:
    v1 = te1_raw[metric]
    v2 = te2_raw[metric]
    s1 = f"{v1:+.3f}" if not pd.isna(v1) else "  N/D "
    s2 = f"{v2:+.3f}" if not pd.isna(v2) else "  N/D "
    print(f"{metric:<22} {s1:<18} {s2:<18}")
print()

# ── RADAR CHART ───────────────────────────────────────────────────────────────
N = len(METRIC_KEYS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def make_radar_values(te_norm_row):
    vals = [float(te_norm_row[m]) for m in METRIC_KEYS]
    vals += vals[:1]
    return vals

v1_radar = make_radar_values(te1_norm)
v2_radar = make_radar_values(te2_norm)

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
    v1_actual = te1_raw[metric]
    v2_actual = te2_raw[metric]
    s1 = f"{v1_actual:+.2f}" if not pd.isna(v1_actual) else "N/D"
    s2 = f"{v2_actual:+.2f}" if not pd.isna(v2_actual) else "N/D"
    label = f"{metric}\n{te1_name.split('.')[-1].strip()}: {s1}\n{te2_name.split('.')[-1].strip()}: {s2}"
    xticklabels.append(label)

ax.set_xticklabels(xticklabels, color=FG, fontsize=7.5, ha="center")

ax.plot(angles, v1_radar, color=TE1_COLOR, linewidth=2.2, zorder=4, label=te1_name)
ax.fill(angles, v1_radar, color=TE1_COLOR, alpha=0.20, zorder=3)

ax.plot(angles, v2_radar, color=TE2_COLOR, linewidth=2.2, zorder=4, label=te2_name)
ax.fill(angles, v2_radar, color=TE2_COLOR, alpha=0.20, zorder=3)

ref_ring = [0.5] * (N + 1)
ax.plot(angles, ref_ring, color=GRID, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)

ax.scatter(angles[:-1], v1_radar[:-1], color=TE1_COLOR, s=30, zorder=5)
ax.scatter(angles[:-1], v2_radar[:-1], color=TE2_COLOR, s=30, zorder=5)

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
safe_te1 = te1_name.split(".")[-1].strip().replace(" ", "_")
safe_te2 = te2_name.split(".")[-1].strip().replace(" ", "_")

fig.text(0.5, 0.97, f"{te1_name} vs {te2_name}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         f"Comparación radar — 6 dimensiones | Normalizadas entre TEs con ≥{MIN_TARGETS} objetivos",
         ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data + stats_player  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = f"comparador_{safe_te1}_{safe_te2}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
