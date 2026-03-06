# DatoSemana.py
# El Dato de la Semana (OUTLIER): detecta automáticamente el rendimiento semanal más extremo
# Descarga directa desde nflverse. Firma @CuartayDato.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
SEASON    = 2025
URL       = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
LOGOS_DIR = "logos"

# Estilo
BG      = "#0f1115"
FG      = "#EDEDED"
GRID    = "#2a2f3a"
DPI     = 200
FIGSIZE = (12, 9)
RYG     = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

HARD_PENALTY = {"NYJ": 4.5}

# ---------- Utilidades ----------
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def robust_zscores(s: pd.Series) -> pd.Series:
    """Z-score robusto basado en mediana y MAD. Fallback a z-score estándar si MAD=0."""
    s = s.dropna()
    if s.empty:
        return s
    med = s.median()
    mad = (s - med).abs().median()
    if mad and mad > 0:
        rz = 0.67448975 * (s - med) / mad
        return rz
    # Fallback
    std = s.std(ddof=0)
    if std and std > 0:
        return (s - s.mean()) / std
    return pd.Series(np.zeros(len(s)), index=s.index)

def logo_image(team, base_zoom=0.055):
    """Carga logo y ajusta zoom por aspecto."""
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

# ---------- Métricas semanales ----------
def metric_series(dfw: pd.DataFrame, key: str):
    """
    Devuelve:
      series: pd.Series index=Team, value=metric
      title: str
      higher_is_better: bool
      fmt: str
      min_plays: int (filtro mínimo)
      count_by_team: pd.Series para info de jugadas/intent.
    """
    if key == "off_epa":
        sub = dfw[dfw["play_type"].isin(["run","pass"]) & dfw["posteam"].notna()]
        cnt = sub.groupby("posteam").size()
        s = sub.groupby("posteam")["epa"].mean()
        return s, "EPA/jugada ofensivo (semana)", True, "{:+.3f}", 25, cnt

    if key == "off_rush":
        sub = dfw[(dfw["play_type"]=="run") & dfw["posteam"].notna()]
        cnt = sub.groupby("posteam").size()
        s = sub.groupby("posteam")["epa"].mean()
        return s, "EPA/carrera ofensivo (semana)", True, "{:+.3f}", 12, cnt

    if key == "off_pass":
        sub = dfw[(dfw["play_type"]=="pass") & dfw["posteam"].notna()]
        cnt = sub.groupby("posteam").size()
        s = sub.groupby("posteam")["epa"].mean()
        return s, "EPA/pase ofensivo (semana)", True, "{:+.3f}", 18, cnt

    if key == "def_epa_allowed":
        sub = dfw[dfw["play_type"].isin(["run","pass"]) & dfw["defteam"].notna()]
        cnt = sub.groupby("defteam").size()
        s = sub.groupby("defteam")["epa"].mean()
        return s, "EPA/jugada permitido (semana)", False, "{:+.3f}", 25, cnt

    if key == "def_rush_allowed":
        sub = dfw[(dfw["play_type"]=="run") & dfw["defteam"].notna()]
        cnt = sub.groupby("defteam").size()
        s = sub.groupby("defteam")["epa"].mean()
        return s, "EPA/carrera permitido (semana)", False, "{:+.3f}", 12, cnt

    if key == "def_pass_allowed":
        sub = dfw[(dfw["play_type"]=="pass") & dfw["defteam"].notna()]
        cnt = sub.groupby("defteam").size()
        s = sub.groupby("defteam")["epa"].mean()
        return s, "EPA/pase permitido (semana)", False, "{:+.3f}", 18, cnt

    if key == "st_epa":
        st_types = {"kickoff","kickoff_return","punt","punt_return","field_goal","extra_point"}
        sub = dfw[dfw["play_type"].isin(st_types) & dfw["posteam"].notna()]
        cnt = sub.groupby("posteam").size()
        s = sub.groupby("posteam")["epa"].mean()
        return s, "EPA/jugada equipos especiales (semana)", True, "{:+.3f}", 6, cnt

    if key == "fg_pct":
        fg_col = "field_goal_result" if "field_goal_result" in dfw.columns else ("fg_result" if "fg_result" in dfw.columns else None)
        if fg_col is None:
            return pd.Series(dtype=float), "FG% (semana)", True, "{:.1f}%", 3, pd.Series(dtype=int)
        fg = dfw[(dfw["play_type"]=="field_goal") & dfw["posteam"].notna()]
        cnt = fg.groupby("posteam").size()
        if fg.empty:
            return pd.Series(dtype=float), "FG% (semana)", True, "{:.1f}%", 3, cnt
        made = fg[fg_col].astype(str).str.lower().eq("made")
        s = made.groupby(fg["posteam"]).mean().mul(100)
        return s, "FG% (semana)", True, "{:.1f}%", 3, cnt

    raise ValueError("key no reconocida")

# ---------- Plot ----------
def plot_outlier(series, title, week, higher_is_better, fmt, counts, out_idx, outfile):
    # Preparación
    s = series.dropna()
    # Orden (mejor arriba si higher_is_better)
    s = s.sort_values(ascending=not higher_is_better)
    teams = s.index.tolist(); vals = s.values

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)

    y = np.arange(len(s))
    ranks = np.argsort(np.argsort(vals)) / max(len(vals) - 1, 1)
    if not higher_is_better:
        ranks = 1.0 - ranks
    base_colors = [RYG(p) for p in ranks]

    # Límites asimétricos
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    rng = max(vmax - vmin, 1e-6)
    ax.set_xlim(vmin - 0.06*rng, vmax + 0.12*rng)

    # Barras
    bars = ax.barh(y, vals, color=base_colors, height=0.56, edgecolor="none")

    # Outlier: contorno y anotación
    if out_idx in s.index:
        idx = teams.index(out_idx)
        bars[idx].set_edgecolor("#ffffff")
        bars[idx].set_linewidth(2.2)

    # Texto de valores
    xr = ax.get_xlim()[1]
    for yy, v, t in zip(y, vals, teams):
        ax.text(min(v + 0.006*rng, xr - 0.01*rng), yy, fmt.format(v),
                va="center", ha="left", fontsize=10, color=FG)

    # Logos a la izquierda
    xmin, xmax = ax.get_xlim()
    x_logo = xmin - 0.075*(xmax - xmin)
    ax.set_yticks([])
    for yy, team in zip(y, teams):
        im = logo_image(team, base_zoom=0.035)
        if im is not None:
            ab = AnnotationBbox(im, (x_logo, yy), frameon=False, xycoords=("data","data"))
            ax.add_artist(ab)
        else:
            ax.text(xmin, yy, team, va="center", ha="left", fontsize=10, color=FG)

    # Título y subtítulo
    fig.text(0.5, 0.97, f"El Dato de la Semana {week}  |  NFL {SEASON}",
             ha="center", va="top", fontsize=18, fontweight="bold", color=FG)
    fig.text(0.5, 0.92, title,
             ha="center", va="top", fontsize=11, color="#888888", fontstyle="italic")

    # Outlier destacado
    out_val = series.loc[out_idx]
    plays = int(counts.get(out_idx, 0)) if isinstance(counts, pd.Series) else 0
    detalle = f"Outlier: {out_idx}  {fmt.format(out_val)}"
    if plays:
        detalle += f"  ({plays} jugadas)"
    ax.text(0.01, 0.02, detalle, transform=ax.transAxes,
            fontsize=10, color="#B9BDC7")

    # Ejes limpios
    ax.grid(axis="x", linestyle="--", alpha=0.25, color=GRID)
    ax.axvline(0, color=GRID, linewidth=1)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    # Fuente
    fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}  ·  Solo pases y carreras",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")

    # Firma
    ax.text(0.99, 0.02, "@CuartayDato", transform=ax.transAxes,
            ha="right", va="bottom", color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

    plt.savefig(outfile, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {outfile}")

# ---------- Main ----------
if __name__ == "__main__":
    week_in = input("Semana (número, p.ej. 5): ").strip()
    try:
        week = int(week_in)
    except:
        raise SystemExit("Semana inválida.")

    print(f"Descargando datos NFL {SEASON}...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    if "week" not in df.columns:
        raise SystemExit("El dataset no tiene columna 'week' (revisa la fuente).")

    dfw = df[df["week"] == week].copy()
    if dfw.empty:
        raise SystemExit(f"No hay jugadas para la semana {week}.")

    # Lista de métricas a evaluar para outliers
    keys = [
        "off_epa", "off_rush", "off_pass",
        "def_epa_allowed", "def_rush_allowed", "def_pass_allowed",
        "st_epa", "fg_pct"
    ]

    best = None  # (abs_z, key, series_filtrada, title, hib, fmt, counts, out_team)
    for key in keys:
        try:
            s_raw, title, hib, fmt, min_plays, counts = metric_series(dfw, key)
        except Exception:
            continue
        if s_raw.empty:
            continue
        # filtro por mínimo volumen
        valid_teams = counts[counts >= min_plays].index
        s = s_raw[s_raw.index.isin(valid_teams)].dropna()
        if s.empty or len(s) < 6:
            continue
        rz = robust_zscores(s)
        if rz.empty:
            continue
        # elegir el más extremo por |z|
        out_team = rz.abs().idxmax()
        abs_z = float(rz.loc[out_team])
        if (best is None) or (abs(abs_z) > abs(best[0])):
            best = (abs_z, key, s, title, hib, fmt, counts, out_team)

    if best is None:
        raise SystemExit("No se pudo determinar un outlier con suficiente volumen de jugadas.")

    abs_z, key, series, title, hib, fmt, counts, out_team = best
    print(f"Outlier detectado -> {out_team} en '{title}' (|z|={abs(abs_z):.2f})")

    outfile = f"dato_semana_outlier_week{week}.png"
    plot_outlier(series, title, week, hib, fmt, counts, out_team, outfile)
