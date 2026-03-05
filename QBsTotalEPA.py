# scatter_qb_total_epa_rz_vs_3rd.py
# QBs 2025 — EPA del QB (pase + carrera)
# Eje X: EPA en Red Zone (yardline_100 <= 20)
# Eje Y: EPA en 3º down (down == 3)
# Incluye jugadas de pase y carrera del propio QB.
# Estilo Cuarta y Dato (oscuro) + firma @CuartayDato.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv.gz"

# -------- Config --------
MIN_QB_PLAYS_RZ   = 8
MIN_QB_PLAYS_3RD  = 8

BG = "#0f1115"
FG = "#EDEDED"
GRID = "#2a2f3a"
DPI = 240

# Paleta de color (igual que otros .py)
cmap = LinearSegmentedColormap.from_list("r2g", ["#ff6b6b", "#ffd166", "#06d6a0"])

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

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
        "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "text.color": FG, "grid.color": GRID, "font.size": 10,
    })

def scatter_with_labels(x, y, labels, size, title, xlabel, ylabel, outfile):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=DPI)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # normalizar tamaño
    size = np.array(size, dtype=float)
    if len(size) and np.nanmax(size) > 0:
        s_norm = 30 + 170 * (size - np.nanmin(size)) / (np.nanmax(size) - np.nanmin(size) + 1e-9)
    else:
        s_norm = np.ones_like(size) * 80

    sc = ax.scatter(x, y, s=s_norm, alpha=0.88, edgecolor="none", c=x+y, cmap=cmap)

    # etiquetas
    for xi, yi, lab in zip(x, y, labels):
        if pd.isna(xi) or pd.isna(yi) or not lab:
            continue
        ax.text(xi, yi, lab, fontsize=8.5, ha="center", va="center", color=FG, alpha=0.9)

    # cuadrícula
    ax.grid(True, linestyle="--", alpha=0.35)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.set_title(title, fontsize=16, pad=10, color=FG, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, color=FG)
    ax.set_ylabel(ylabel, fontsize=12, color=FG)

    # Firma
    ax.text(
        0.99, 0.02, "@CuartayDato", fontsize=9.5, color="#A0A3AA",
        ha="right", va="bottom", transform=ax.transAxes, alpha=0.8
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✅ Guardado: {outfile}")

# -------- Main --------
def main():
    print("Descargando play-by-play 2025…")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    to_num(df, ["epa", "down", "yardline_100"])

    pass_type = "pass"; run_type = "run"
    passer_id_col = pick_col(df, "passer_player_id", "passer_id")
    rusher_id_col = pick_col(df, "rusher_player_id", "rusher_id")
    passer_name_col = pick_col(df, "passer", "passer_player_name")
    rusher_name_col = pick_col(df, "rusher", "rusher_player_name")

    base = df[df["posteam"].notna() & df["play_type"].isin([pass_type, run_type])].copy()
    pass_df = base[base["play_type"].eq(pass_type)].copy()

    qb_ids = set(pass_df[passer_id_col].dropna().unique()) if passer_id_col else set()
    id2name = {}
    if passer_id_col and passer_name_col:
        tmp = pass_df.dropna(subset=[passer_id_col, passer_name_col])
        if not tmp.empty:
            id2name = tmp.groupby(passer_id_col)[passer_name_col].agg(lambda s: s.value_counts().idxmax()).to_dict()

    rows = []
    # Pases
    for _, r in pass_df.iterrows():
        qb_key = r.get(passer_id_col) if passer_id_col else r.get(passer_name_col)
        if pd.isna(qb_key):
            continue
        rows.append({
            "qb_key": qb_key,
            "qb_name": id2name.get(qb_key, r.get(passer_name_col, "")),
            "epa": r["epa"],
            "is_rz": bool(pd.notna(r.get("yardline_100")) and r["yardline_100"] <= 20),
            "is_3rd": bool(r.get("down") == 3)
        })

    # Carreras
    run_df = base[base["play_type"].eq(run_type)].copy()
    for _, r in run_df.iterrows():
        if rusher_id_col and len(qb_ids) > 0:
            is_qb = r.get(rusher_id_col) in qb_ids
            qb_key = r.get(rusher_id_col)
            qb_name = id2name.get(qb_key, r.get(rusher_name_col, ""))
        else:
            if passer_name_col and rusher_name_col:
                is_qb = r.get(rusher_name_col) in set(pass_df[passer_name_col].dropna().unique())
                qb_key = r.get(rusher_name_col)
                qb_name = r.get(rusher_name_col, "")
            else:
                is_qb = False
                qb_key = None
                qb_name = ""
        if not is_qb or pd.isna(qb_key):
            continue
        rows.append({
            "qb_key": qb_key,
            "qb_name": qb_name,
            "epa": r["epa"],
            "is_rz": bool(pd.notna(r.get("yardline_100")) and r["yardline_100"] <= 20),
            "is_3rd": bool(r.get("down") == 3)
        })

    qb_plays = pd.DataFrame.from_records(rows)
    if qb_plays.empty:
        raise SystemExit("No se han podido construir jugadas por QB.")

    # Métricas
    rz = qb_plays[qb_plays["is_rz"]].groupby("qb_key").agg(rz_epa=("epa","mean"), rz_plays=("epa","size"))
    d3 = qb_plays[qb_plays["is_3rd"]].groupby("qb_key").agg(d3_epa=("epa","mean"), d3_plays=("epa","size"))
    qbs = rz.join(d3, how="outer").fillna(0)

    name_map = qb_plays.dropna(subset=["qb_key","qb_name"]).groupby("qb_key")["qb_name"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    qbs["label"] = [short_name(name_map.get(k, str(k))) for k in qbs.index]
    qbs = qbs[(qbs["rz_plays"] >= MIN_QB_PLAYS_RZ) & (qbs["d3_plays"] >= MIN_QB_PLAYS_3RD)].copy()

    if qbs.empty:
        raise SystemExit("Tras aplicar filtros de volumen no hay QBs suficientes.")

    outfile = "scatter_QB_totalEPA_RZ_vs_3rd.png"
    scatter_with_labels(
        x=qbs["rz_epa"].values,
        y=qbs["d3_epa"].values,
        labels=qbs["label"].values,
        size=(qbs["rz_plays"] + qbs["d3_plays"]).values,
        title="QBs 2025 — EPA del QB (pase + carrera)\nRedZone (X) vs 3º down (Y)",
        xlabel="EPA/jugada en Red Zone",
        ylabel="EPA/jugada en 3º down",
        outfile=outfile
    )

if __name__ == "__main__":
    main()
