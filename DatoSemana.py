# DatoSemana.py
# El Dato de la Semana (OUTLIER): detecta automáticamente el rendimiento semanal más extremo
# Descarga directa desde nflverse. Firma @CuartayDato.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, salida, season_cli, week_cli, sello
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
SEASON    = season_cli()   # None = auto-detectar última temporada


def excluir_cli():
    """--excluir SF,ATL: equipos que no pueden ser el dato (Luis, 29-sep-2026:
    el ataque de SF ya fue el dato de la semana 2 y salio en los posts del
    lunes). Se descartan como outlier, pero siguen en el grafico."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--excluir", default="")
    args, _ = ap.parse_known_args()
    return {x.strip().upper() for x in args.excluir.split(",") if x.strip()}


EXCLUIR = excluir_cli()
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

# ---------- Plot (estilo docs/estilo-visual.md) ----------
# Frase del titular por metrica: (si es el MEJOR, si es el PEOR). {T} = equipo.
FRASES = {
    "EPA/jugada ofensivo (semana)": ("Ningún ataque sumó tanto por jugada como el de {T}.",
                                     "Ningún ataque restó tanto por jugada como el de {T}."),
    "EPA/carrera ofensivo (semana)": ("Ningún equipo corrió como {T}.",
                                      "Ningún equipo corrió peor que {T}."),
    "EPA/pase ofensivo (semana)": ("Ningún equipo pasó como {T}.",
                                   "Ningún equipo pasó peor que {T}."),
    "EPA/jugada equipos especiales (semana)": ("Nadie sacó tanto de los equipos especiales como {T}.",
                                               "Ningunos especiales costaron tanto como los de {T}."),
    "FG% (semana)": ("Nadie acertó tanto en los field goals como {T}.",
                     "Nadie falló tanto en los field goals como {T}."),
    "EPA/jugada permitido (semana)": ("Ninguna defensa concedió tan poco como la de {T}.",
                                      "Ninguna defensa concedió tanto como la de {T}."),
    "EPA/carrera permitido (semana)": ("Ninguna defensa paró la carrera como la de {T}.",
                                       "Ninguna defensa sufrió tanto contra la carrera como la de {T}."),
    "EPA/pase permitido (semana)": ("Ninguna defensa paró el pase como la de {T}.",
                                    "Ninguna defensa sufrió tanto contra el pase como la de {T}."),
}
ETIQUETA = {k: k.replace(" (semana)", "").replace("EPA/", "EPA por ").replace(" ofensivo", "")
            .replace(" permitido", " permitido").replace("FG%", "Acierto en field goals")
            for k in FRASES}


def nombre_equipo(team):
    """Ciudad para el titular; los equipos que comparten ciudad, por su apodo."""
    try:
        eq = pd.read_parquet("pbp_cache/teams.parquet").set_index("team_abbr")
        if team in ("LA", "LAC", "NYG", "NYJ"):
            return "los " + eq.loc[team, "team_nick"]
        return eq.loc[team, "team_name"].replace(" " + eq.loc[team, "team_nick"], "")
    except Exception:
        return team


def fmt_v(v, title):
    return f"{v:.0f}%" if title.startswith("FG%") else num(v)


def plot_outlier(series, title, week, higher_is_better, fmt, counts, out_idx, outfile):
    s = series.dropna().sort_values(ascending=not higher_is_better)    # el mejor primero
    teams = s.index.tolist()
    n = len(teams)
    rk = teams.index(out_idx) + 1
    T = nombre_equipo(out_idx)
    # "Ninguna..." solo si de verdad es el 1º o el ultimo (el z va encogido por muestra)
    if rk == 1:
        tit = FRASES.get(title, ("", ""))[0].format(T=T)
    elif rk == n:
        tit = FRASES.get(title, ("", ""))[1].format(T=T)
    else:
        tit = ""
    if not tit:
        tit = f"{T}: {fmt_v(s.loc[out_idx], title)} {ETIQUETA.get(title, title).lower()}, {rk}º de {n}."
    tit = tit[0].upper() + tit[1:]

    fig = lienzo()
    antetitulo(fig, f"Dato de la semana {week}")
    y = titular(fig, tit, y=0.925) - 0.03
    y_top = y
    yc = y - 0.04
    poner_logo(fig, out_idx, L + 0.035, yc, zoom=0.024)
    fig.text(L + 0.085, yc, out_idx, family=TXT, weight="bold", size=9,
             color=color_equipo(out_idx), va="center")
    lado = "DEFENSA" if "permitido" in title else ("ESPECIALES" if ("especiales" in title or "FG" in title) else "ATAQUE")
    fig.text(L + 0.15, yc, f"·  {lado}", family=TXT, weight="semibold", size=8.5, color=GRIS1,
             va="center")
    etiqueta(fig, ETIQUETA.get(title, title), L + 0.01, yc - 0.05)
    numero(fig, fmt_v(s.loc[out_idx], title), L + 0.005, yc - 0.11)
    fig.text(L + 0.43, yc - 0.095, f"{rk}º de {n}", family=TXT, weight="bold", size=9,
             color=color_rango(rk, n), va="center")
    plays = int(counts.get(out_idx, 0)) if isinstance(counts, pd.Series) else 0
    otro = teams[1] if rk == 1 else teams[0]
    nota(fig, f"{plays} jugadas · {'el 2º' if rk == 1 else 'el 1º'}, {otro}, "
              f"{fmt_v(s.loc[otro], title)}", L + 0.43, yc - 0.123)

    # Todos los equipos: gris; el protagonista en su color y el siguiente rotulado
    etiqueta(fig, f"Los {n} equipos de la semana", L + 0.01, yc - 0.19)
    alto = yc - 0.215 - 0.17
    ax = estilo_ejes(fig.add_axes([L + 0.01, 0.17, R - L - 0.02, alto]), x=True)
    vals = s.values
    for i, (t, v) in enumerate(zip(teams, vals)):
        yy = n - 1 - i
        prota = t == out_idx
        c = color_equipo(t) if prota else (GRIS1 if t == otro else NEUTRO)
        ax.plot([0, v], [yy, yy], color=c, lw=1.6 if prota else 1.0, solid_capstyle="butt")
        ax.scatter([v], [yy], s=26 if prota else 9, color=c, zorder=3, lw=0)
        if prota or t == otro:
            ax.text(v + (0.012 if v >= 0 else -0.012) * (max(abs(vals)) or 1), yy, t,
                    family=TXT, weight="bold" if prota else "normal", size=7,
                    color=c, va="center", ha="left" if v >= 0 else "right")
    ax.axvline(0, color=GRID, lw=1)
    span = max(abs(vals)) or 1
    ax.set_xlim(min(0, vals.min()) - 0.15 * span, max(0, vals.max()) + 0.15 * span)
    ax.set_ylim(-1, n)
    ax.set_xticks([])
    fig.text(L + 0.01, 0.155, "MEJOR" if not higher_is_better else "PEOR", family=TXT,
             weight="medium", size=6.5, color=GRIS2, va="center")
    fig.text(R - 0.01, 0.155, "PEOR" if not higher_is_better else "MEJOR", family=TXT,
             weight="medium", size=6.5, color=GRIS2, va="center", ha="right")

    tarjeta(fig, 0.13, y_top)
    pie(fig, f"{ETIQUETA.get(title, title)} en la semana {week}",
        f"NFL {SEASON} · semana {week} · nflverse PBP")
    guardar(fig, outfile)

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
        rz_aj = rz_aj[~rz_aj.index.isin(EXCLUIR)]
        if rz_aj.empty:
            continue
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
            rival = s_def.drop(list(EXCLUIR), errors="ignore").idxmin()  # menos EPA permitido = mejor defensa
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
