# DatoSemana.py
# El Dato de la Semana (OUTLIER): detecta automáticamente el rendimiento semanal más extremo
# Descarga directa desde nflverse. Firma @CuartayDato.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, salida, season_cli, week_cli, sello
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
SEASON    = season_cli()   # None = auto-detectar última temporada
LOGOS_DIR = "logos"

# Estilo
BG      = "#0f1115"
FG      = "#EDEDED"
GRID    = "#2a2f3a"
DPI     = 170
FIGSIZE = (12, 9)
RYG     = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])


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

    if key == "st_epa":
        st_types = {"kickoff","kickoff_return","punt","punt_return","field_goal","extra_point"}
        sub = dfw[dfw["play_type"].isin(st_types) & dfw["posteam"].notna()]
        cnt = sub.groupby("posteam").size()
        s = sub.groupby("posteam")["epa"].mean()
        return s, "EPA/jugada equipos especiales (semana)", True, "{:+.3f}", 10, cnt

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

# Cada metrica ofensiva tiene su espejo defensivo: son LAS MISMAS jugadas
# reagrupadas por el equipo que defiende. Por eso no se evaluan como candidatas
# (darian exactamente el mismo z y el desempate iria siempre al ataque); se usan
# para reencuadrar el titular cuando el outlier es un desastre y no una hazaña.
ESPEJO_DEF = {
    "off_epa":  ("EPA/jugada permitido (semana)",  "run/pass"),
    "off_rush": ("EPA/carrera permitido (semana)", "run"),
    "off_pass": ("EPA/pase permitido (semana)",    "pass"),
}


def voltear_a_defensa(dfw, key):
    """Misma metrica agrupada por defteam. Devuelve (serie, titulo, cuentas)."""
    titulo, tipos = ESPEJO_DEF[key]
    filtro = ["run", "pass"] if tipos == "run/pass" else [tipos]
    sub = dfw[dfw["play_type"].isin(filtro) & dfw["defteam"].notna()]
    return sub.groupby("defteam")["epa"].mean(), titulo, sub.groupby("defteam").size()


def penalizacion_muestra(n, k=15):
    """Encoge el z segun el tamaño de muestra del outlier.

    Sin esto, equipos especiales gana un tercio de las semanas: con 6-10 jugadas
    un punt bloqueado parece mas 'extremo' que un ataque brillante sobre 65
    jugadas. Con k=15 una muestra de 8 pesa 0.59 y una de 65 pesa 0.90.
    """
    n = max(int(n), 1)
    return (n / (n + k)) ** 0.5

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

    # Texto de valores (negativos a la izquierda, fuera de la barra)
    xr = ax.get_xlim()[1]
    for yy, v, t in zip(y, vals, teams):
        if v >= 0:
            ax.text(min(v + 0.006*rng, xr - 0.01*rng), yy, fmt.format(v),
                    va="center", ha="left", fontsize=10, color=FG)
        else:
            ax.text(v - 0.006*rng, yy, fmt.format(v),
                    va="center", ha="right", fontsize=10, color=FG)

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
    # Sin coletilla "solo pases y carreras": la métrica puede ser de ST/FG
    fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  {sello(SEASON)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")

    # Firma
    ax.text(0.99, 0.02, "@CuartayDato", transform=ax.transAxes,
            ha="right", va="bottom", color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

    plt.savefig(outfile, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {outfile}")

# ---------- Main ----------
if __name__ == "__main__":
    week_in = str(week_cli() or "") or input("Semana (número, p.ej. 5): ").strip()
    try:
        week = int(week_in)
    except:
        raise SystemExit("Semana inválida.")

    # solo_reg=False: el filtro semanal es del usuario (semanas 19+ = playoffs)
    df, SEASON = cargar_pbp(SEASON, solo_reg=False)

    dfw = df[df["week"] == week].copy()
    if dfw.empty:
        raise SystemExit(f"No hay jugadas para la semana {week}.")

    # Solo metricas ofensivas y de equipos especiales: las defensivas son las
    # mismas jugadas reagrupadas (ver ESPEJO_DEF) y jamas ganarian el desempate.
    keys = ["off_epa", "off_rush", "off_pass", "st_epa", "fg_pct"]

    best = None  # (z, key, series_filtrada, title, hib, fmt, counts, out_team)
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
        # elegir el más extremo por |z|, encogido por tamaño de muestra
        rz_aj = rz * rz.index.map(lambda t: penalizacion_muestra(counts.get(t, 0)))
        out_team = rz_aj.abs().idxmax()
        z = float(rz_aj.loc[out_team])
        if (best is None) or (abs(z) > abs(best[0])):
            best = (z, key, s, title, hib, fmt, counts, out_team)

    if best is None:
        raise SystemExit("No se pudo determinar un outlier con suficiente volumen de jugadas.")

    z, key, series, title, hib, fmt, counts, out_team = best

    # Reencuadre: un ataque hundido es, contado desde el otro lado, una gran
    # actuacion defensiva — y casi siempre es la mejor historia de las dos.
    if z < 0 and key in ESPEJO_DEF:
        s_def, title_def, cnt_def = voltear_a_defensa(dfw, key)
        s_def = s_def.dropna()
        if len(s_def) >= 6:
            rival = s_def.idxmin()          # menos EPA permitido = mejor defensa
            print(f"Outlier -> {out_team} hundido en '{title}' (z={z:+.2f})")
            print(f"Reencuadrado como defensa: {rival} en '{title_def}'")
            series, title, hib, fmt, counts, out_team = (
                s_def, title_def, False, fmt, cnt_def, rival)
        else:
            print(f"Outlier detectado -> {out_team} en '{title}' (z={z:+.2f})")
    else:
        print(f"Outlier detectado -> {out_team} en '{title}' (z={z:+.2f})")

    outfile = salida(f"dato_semana_outlier_{SEASON}.png", SEASON, week)
    plot_outlier(series, title, week, hib, fmt, counts, out_team, outfile)
