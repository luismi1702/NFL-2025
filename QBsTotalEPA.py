# QBsTotalEPA.py
# QBs — EPA/jugada en Red Zone (X) vs 3er down (Y)
# Logo de equipo en cada punto + nombre debajo. Estilo Cuarta y Dato.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Config ===
SEASON           = 2025
MIN_WEEK         = 1
MAX_WEEK         = 18
MIN_QB_PLAYS_RZ  = 50   # mínimo jugadas en red zone
MIN_QB_PLAYS_3RD = 50   # mínimo jugadas en 3er down

URL       = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
OUT       = f"scatter_QB_totalEPA_RZ_vs_3rd_{SEASON}.png"
LOGOS_DIR = "logos"

BG           = "#0f1115"
FG           = "#EDEDED"
GRID         = "#2a2f3a"
DPI          = 200
HARD_PENALTY = {"NYJ": 4.5}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def short_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    parts = name.replace("-", " ").split()
    if len(parts) == 1:
        return parts[0][:14]
    return (parts[0][:1] + ". " + parts[-1])[:16]

def load_logo(team, base_zoom=0.030):
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

# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot_qb_scatter(qbs_df, team_map, title, xlabel, ylabel, outfile):
    fig, ax = plt.subplots(figsize=(13, 9), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    plt.setp(ax.get_xticklabels(), color=FG)
    plt.setp(ax.get_yticklabels(), color=FG)

    # Márgenes del eje
    x_vals = qbs_df["rz_epa"].values
    y_vals = qbs_df["d3_epa"].values
    x_pad = (x_vals.max() - x_vals.min()) * 0.14
    y_pad = (y_vals.max() - y_vals.min()) * 0.14
    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

    # Líneas de referencia
    ax.axhline(0, color=GRID, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
    ax.axvline(0, color=GRID, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
    ax.grid(True, linestyle="--", alpha=0.12, color=GRID, zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Etiquetas de cuadrante
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    xm = (x_hi - x_lo) * 0.03
    ym = (y_hi - y_lo) * 0.03
    q_kw = dict(fontsize=8.5, alpha=0.30, color=FG, fontstyle="italic")
    ax.text(x_hi - xm, y_hi - ym, "Élite en ambas",        ha="right", va="top",    **q_kw)
    ax.text(x_lo + xm, y_hi - ym, "Bueno en 3ro / Malo RZ", ha="left",  va="top",    **q_kw)
    ax.text(x_hi - xm, y_lo + ym, "Bueno en RZ / Malo 3ro", ha="right", va="bottom", **q_kw)
    ax.text(x_lo + xm, y_lo + ym, "Peor en ambas",          ha="left",  va="bottom", **q_kw)

    # Offset vertical para el nombre (en unidades de datos)
    y_range      = y_hi - y_lo
    label_offset = y_range * 0.038

    # Logos + nombres
    for qb_key, row in qbs_df.iterrows():
        x    = row["rz_epa"]
        y    = row["d3_epa"]
        name = row["label"]
        team = team_map.get(qb_key, "")

        logo = load_logo(team, base_zoom=0.030)
        if logo:
            ab = AnnotationBbox(logo, (x, y),
                                frameon=False, zorder=3,
                                box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        else:
            ax.scatter(x, y, s=90, color="#888888", zorder=3, alpha=0.8)

        ax.text(x, y - label_offset, name,
                ha="center", va="top",
                fontsize=7.5, color=FG, alpha=0.88, zorder=4)

    # Ejes y títulos
    ax.set_xlabel(xlabel, fontsize=11, color=FG, labelpad=7)
    ax.set_ylabel(ylabel, fontsize=11, color=FG, labelpad=7)
    ax.set_title(title, fontsize=15, pad=12, color=FG, fontweight="bold")

    week_range = f"S{MIN_WEEK}-S{MAX_WEEK}" if MAX_WEEK < 18 else "Temporada completa"
    fig.text(0.5, 0.01,
             f"Fuente: nflverse-data  ·  mín. {MIN_QB_PLAYS_RZ} jugadas en RZ y {MIN_QB_PLAYS_3RD} en 3er down  ·  {week_range}",
             ha="center", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    ax.text(0.99, 0.02, "@CuartayDato", fontsize=9, color="#888888",
            ha="right", va="bottom", transform=ax.transAxes, alpha=0.85, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {outfile}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Descargando play-by-play {SEASON} semanas {MIN_WEEK}-{MAX_WEEK}...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    to_num(df, ["epa", "down", "yardline_100", "week"])

    passer_id_col   = pick_col(df, "passer_player_id", "passer_id")
    rusher_id_col   = pick_col(df, "rusher_player_id", "rusher_id")
    passer_name_col = pick_col(df, "passer", "passer_player_name")

    base = df[
        df["week"].between(MIN_WEEK, MAX_WEEK) &
        df["posteam"].notna() &
        df["play_type"].isin(["pass", "run"])
    ].copy()

    pass_df = base[base["play_type"] == "pass"].dropna(subset=[passer_id_col]).copy()
    pass_df["qb_key"]  = pass_df[passer_id_col]
    pass_df["qb_name"] = pass_df[passer_name_col]

    id2name = (pass_df.dropna(subset=[passer_id_col, passer_name_col])
               .groupby(passer_id_col)[passer_name_col]
               .agg(lambda s: s.value_counts().idxmax()))

    # Equipo principal de cada QB
    team_map = (pass_df.dropna(subset=["qb_key", "posteam"])
                .groupby("qb_key")["posteam"]
                .agg(lambda s: s.value_counts().idxmax())
                .to_dict())

    # Carreras de QB
    qb_ids  = set(pass_df[passer_id_col].dropna().unique())
    run_df  = base[base["play_type"] == "run"].copy()
    qb_runs = run_df[run_df[rusher_id_col].isin(qb_ids)].copy()
    qb_runs["qb_key"]  = qb_runs[rusher_id_col]
    qb_runs["qb_name"] = qb_runs[rusher_id_col].map(id2name)

    cols     = ["qb_key", "qb_name", "epa", "yardline_100", "down"]
    qb_plays = pd.concat([pass_df[cols], qb_runs[cols]], ignore_index=True)
    qb_plays = qb_plays.dropna(subset=["qb_key", "epa"])
    qb_plays["is_rz"]  = qb_plays["yardline_100"] <= 20
    qb_plays["is_3rd"] = qb_plays["down"] == 3

    if qb_plays.empty:
        raise SystemExit("No se han podido construir jugadas por QB.")

    rz  = qb_plays[qb_plays["is_rz"]].groupby("qb_key").agg(rz_epa=("epa","mean"), rz_plays=("epa","size"))
    d3  = qb_plays[qb_plays["is_3rd"]].groupby("qb_key").agg(d3_epa=("epa","mean"), d3_plays=("epa","size"))
    qbs = rz.join(d3, how="inner")
    qbs = qbs[(qbs["rz_plays"] >= MIN_QB_PLAYS_RZ) & (qbs["d3_plays"] >= MIN_QB_PLAYS_3RD)].copy()

    name_map = (qb_plays.dropna(subset=["qb_key","qb_name"])
                .groupby("qb_key")["qb_name"]
                .agg(lambda s: s.value_counts().idxmax()))
    qbs["label"] = [short_name(name_map.get(k, str(k))) for k in qbs.index]

    if qbs.empty:
        raise SystemExit("Tras aplicar filtros de volumen no hay QBs suficientes.")

    print(f"QBs en el scatter: {len(qbs)}")
    week_label = f"S{MIN_WEEK}-S{MAX_WEEK}" if MAX_WEEK < 18 else str(SEASON)
    plot_qb_scatter(
        qbs_df   = qbs,
        team_map = team_map,
        title    = f"QBs NFL {week_label} — EPA Red Zone vs 3er down",
        xlabel   = "EPA/jugada en Red Zone",
        ylabel   = "EPA/jugada en 3er down",
        outfile  = OUT,
    )

if __name__ == "__main__":
    main()
