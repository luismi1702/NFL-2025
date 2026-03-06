# QBsTotalEPA.py
# QBs — EPA del QB (pase + carrera)
# Eje X: EPA en Red Zone (yardline_100 <= 20)
# Eje Y: EPA en 3º down (down == 3)
# Incluye jugadas de pase y carrera del propio QB.
# Estilo Cuarta y Dato (oscuro) + firma @CuartayDato.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
SEASON           = 2025
MIN_WEEK         = 1
MAX_WEEK         = 18    # ajustar para análisis mid-season
MIN_QB_PLAYS_RZ  = 15    # mínimo jugadas en red zone para incluir al QB
MIN_QB_PLAYS_3RD = 15    # mínimo jugadas en 3º down

URL  = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
OUT  = f"scatter_QB_totalEPA_RZ_vs_3rd_{SEASON}.png"

BG   = "#0f1115"
FG   = "#EDEDED"
GRID = "#2a2f3a"
DPI  = 200

cmap = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

# -------- Utilidades --------
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
    first = parts[0]; last = parts[-1]
    return (first[:1] + ". " + last)[:16]

def scatter_with_labels(x, y, labels, size, title, xlabel, ylabel, outfile):
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

    # Tamaño de punto proporcional al volumen de jugadas
    size = np.array(size, dtype=float)
    if len(size) and np.nanmax(size) > 0:
        s_norm = 30 + 170 * (size - np.nanmin(size)) / (np.nanmax(size) - np.nanmin(size) + 1e-9)
    else:
        s_norm = np.ones_like(size) * 80

    ax.scatter(x, y, s=s_norm, alpha=0.88, edgecolor="none", c=x + y, cmap=cmap)

    for xi, yi, lab in zip(x, y, labels):
        if pd.isna(xi) or pd.isna(yi) or not lab:
            continue
        ax.text(xi, yi, lab, fontsize=8.5, ha="center", va="center", color=FG, alpha=0.9)

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.2, color=GRID)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.set_title(title, fontsize=15, pad=10, color=FG, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11, color=FG, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=11, color=FG, labelpad=6)

    week_range = f"S{MIN_WEEK}-S{MAX_WEEK}" if MAX_WEEK < 18 else "Temporada completa"
    fig.text(0.5, 0.01,
             f"Fuente: nflverse-data  ·  min. {MIN_QB_PLAYS_RZ} jugadas en RZ y {MIN_QB_PLAYS_3RD} en 3er down  ·  {week_range}",
             ha="center", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")

    ax.text(0.99, 0.02, "@CuartayDato", fontsize=9, color="#888888",
            ha="right", va="bottom", transform=ax.transAxes, alpha=0.85, fontstyle="italic")

    plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {outfile}")

# -------- Main --------
def main():
    print(f"Descargando play-by-play {SEASON} semanas {MIN_WEEK}-{MAX_WEEK}...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    to_num(df, ["epa", "down", "yardline_100", "week"])

    passer_id_col   = pick_col(df, "passer_player_id", "passer_id")
    rusher_id_col   = pick_col(df, "rusher_player_id", "rusher_id")
    passer_name_col = pick_col(df, "passer", "passer_player_name")
    rusher_name_col = pick_col(df, "rusher", "rusher_player_name")

    base = df[
        df["week"].between(MIN_WEEK, MAX_WEEK) &
        df["posteam"].notna() &
        df["play_type"].isin(["pass", "run"])
    ].copy()

    pass_df = base[base["play_type"] == "pass"].dropna(subset=[passer_id_col]).copy()
    pass_df["qb_key"]  = pass_df[passer_id_col]
    pass_df["qb_name"] = pass_df[passer_name_col]

    # Mapa id → nombre más frecuente
    id2name = (pass_df.dropna(subset=[passer_id_col, passer_name_col])
               .groupby(passer_id_col)[passer_name_col]
               .agg(lambda s: s.value_counts().idxmax()))

    # Carreras de QB (vectorizado: rusher_id está en el set de passers)
    qb_ids  = set(pass_df[passer_id_col].dropna().unique())
    run_df  = base[base["play_type"] == "run"].copy()
    qb_runs = run_df[run_df[rusher_id_col].isin(qb_ids)].copy()
    qb_runs["qb_key"]  = qb_runs[rusher_id_col]
    qb_runs["qb_name"] = qb_runs[rusher_id_col].map(id2name)

    cols = ["qb_key", "qb_name", "epa", "yardline_100", "down"]
    qb_plays = pd.concat([pass_df[cols], qb_runs[cols]], ignore_index=True)
    qb_plays = qb_plays.dropna(subset=["qb_key", "epa"])
    qb_plays["is_rz"]  = qb_plays["yardline_100"] <= 20
    qb_plays["is_3rd"] = qb_plays["down"] == 3

    if qb_plays.empty:
        raise SystemExit("No se han podido construir jugadas por QB.")

    # Métricas por QB
    rz  = qb_plays[qb_plays["is_rz"]].groupby("qb_key").agg(rz_epa=("epa","mean"), rz_plays=("epa","size"))
    d3  = qb_plays[qb_plays["is_3rd"]].groupby("qb_key").agg(d3_epa=("epa","mean"), d3_plays=("epa","size"))
    qbs = rz.join(d3, how="inner")  # solo QBs con datos en ambas métricas
    qbs = qbs[(qbs["rz_plays"] >= MIN_QB_PLAYS_RZ) & (qbs["d3_plays"] >= MIN_QB_PLAYS_3RD)].copy()

    name_map = (qb_plays.dropna(subset=["qb_key","qb_name"])
                .groupby("qb_key")["qb_name"]
                .agg(lambda s: s.value_counts().idxmax()))
    qbs["label"] = [short_name(name_map.get(k, str(k))) for k in qbs.index]

    if qbs.empty:
        raise SystemExit("Tras aplicar filtros de volumen no hay QBs suficientes.")

    print(f"QBs en el scatter: {len(qbs)}")
    week_label = f"S{MIN_WEEK}-S{MAX_WEEK}" if MAX_WEEK < 18 else f"{SEASON}"
    scatter_with_labels(
        x=qbs["rz_epa"].values,
        y=qbs["d3_epa"].values,
        labels=qbs["label"].values,
        size=(qbs["rz_plays"] + qbs["d3_plays"]).values,
        title=f"QBs NFL {week_label} — EPA Red Zone (X) vs 3er down (Y)",
        xlabel="EPA/jugada en Red Zone",
        ylabel="EPA/jugada en 3er down",
        outfile=OUT
    )

if __name__ == "__main__":
    main()
