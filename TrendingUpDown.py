# TrendingUpDown.py
# Trending Up / Down — unidad (Ataque o Defensa) con mayor cambio para un equipo.
# Gráfico HORIZONTAL: barras "Previas" vs "Últimas 3".
# Descarga directa desde nflverse. Firma @CuartayDato.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Config ===
SEASON = 2025
URL    = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

# Filtros mínimos (evitan ruido)
MIN_PLAYS_OFF_LAST3 = 40
MIN_PLAYS_OFF_PREV  = 60
MIN_PLAYS_DEF_LAST3 = 40
MIN_PLAYS_DEF_PREV  = 60

# Estilo
BG        = "#0f1115"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
GREEN     = "#06d6a0"
RED       = "#ef476f"
DPI       = 200
LOGOS_DIR = "logos"

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

HARD_PENALTY = {"NYJ": 4.5}

def load_logo_image(team, base_zoom=0.10):
    """Carga logo y devuelve OffsetImage con corrección de aspecto."""
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

def main():
    print("Descargando datos 2025 de nflverse…")
    df = pd.read_csv(URL, low_memory=False, compression="infer")

    if "week" not in df.columns:
        raise SystemExit("El dataset no tiene columna 'week'.")
    to_num(df, ["week", "epa"])

    base = df[df["play_type"].isin(["pass", "run"])].copy()
    base = base[base["posteam"].notna() & base["defteam"].notna()]
    if base.empty:
        raise SystemExit("No hay jugadas válidas (pass/run con equipos definidos).")

    # Últimas 3 semanas disponibles
    max_week = int(base["week"].max())
    last3_weeks = [w for w in [max_week-2, max_week-1, max_week] if w >= base["week"].min()]
    # ✅ corrección del chequeo de semanas
    if len(last3_weeks) < 1 or base["week"].isin(last3_weeks).sum() == 0:
        raise SystemExit("No hay suficientes semanas para calcular últimas 3.")
    prev_mask = base["week"] < min(last3_weeks)

    print(f"Semanas analizadas: previas < {min(last3_weeks)}, últimas {sorted(last3_weeks)}")

    # ---- OFENSIVO (EPA/play por posteam) ----
    off_last3 = base[base["week"].isin(last3_weeks)].groupby("posteam").agg(
        epa_mean=("epa", "mean"), plays=("epa","size")
    ).rename(columns={"epa_mean":"off_epa_last3","plays":"off_plays_last3"})
    off_prev = base[prev_mask].groupby("posteam").agg(
        epa_mean=("epa", "mean"), plays=("epa","size")
    ).rename(columns={"epa_mean":"off_epa_prev","plays":"off_plays_prev"})
    off = off_last3.join(off_prev, how="inner")
    off = off[(off["off_plays_last3"] >= MIN_PLAYS_OFF_LAST3) & (off["off_plays_prev"] >= MIN_PLAYS_OFF_PREV)]
    off["mejora"] = off["off_epa_last3"] - off["off_epa_prev"]  # + = mejora ofensiva
    off["unidad"] = "Ataque"
    off = off.reset_index().rename(columns={"posteam":"Equipo"})

    # ---- DEFENSIVO (EPA/play permitido por defteam) ----
    def_last3 = base[base["week"].isin(last3_weeks)].groupby("defteam").agg(
        epa_mean=("epa", "mean"), plays=("epa","size")
    ).rename(columns={"epa_mean":"def_epa_allowed_last3","plays":"def_plays_last3"})
    def_prev = base[prev_mask].groupby("defteam").agg(
        epa_mean=("epa", "mean"), plays=("epa","size")
    ).rename(columns={"epa_mean":"def_epa_allowed_prev","plays":"def_plays_prev"})
    deff = def_last3.join(def_prev, how="inner")
    deff = deff[(deff["def_plays_last3"] >= MIN_PLAYS_DEF_LAST3) & (deff["def_plays_prev"] >= MIN_PLAYS_DEF_PREV)]
    deff["mejora"] = deff["def_epa_allowed_prev"] - deff["def_epa_allowed_last3"]  # + = mejora defensiva
    deff["unidad"] = "Defensa"
    deff = deff.reset_index().rename(columns={"defteam":"Equipo"})

    combined = pd.concat([off, deff], ignore_index=True, sort=False)
    if combined.empty:
        raise SystemExit("No hay suficientes datos tras aplicar filtros de volumen.")

    combined_sorted = combined.sort_values("mejora", ascending=False)
    top_mejoras = combined_sorted.head(3).copy()
    top_peores  = combined_sorted.tail(3).copy()

    print("\n====== TRENDING UP (Top 3) ======")
    print(top_mejoras[["Equipo","unidad","mejora"]].to_string(index=False))
    print("\n====== TRENDING DOWN (Bottom 3) ======")
    print(top_peores[["Equipo","unidad","mejora"]].to_string(index=False))

    # ----- Selección de equipo -----
    team_choice = input("\nElige equipo para graficar (siglas, p.ej. SF, DAL, NYJ): ").strip().upper()
    t_off = off[off["Equipo"] == team_choice]
    t_def = deff[deff["Equipo"] == team_choice]
    if t_off.empty and t_def.empty:
        print(f"No hay datos suficientes para {team_choice}.")
        return

    # Elegir unidad con mayor cambio absoluto
    cand = []
    if not t_off.empty:
        cand.append(("Ataque", float(t_off["mejora"].iloc[0])))
    if not t_def.empty:
        cand.append(("Defensa", float(t_def["mejora"].iloc[0])))
    unidad_sel, delta = max(cand, key=lambda x: abs(x[1]))

    if unidad_sel == "Ataque":
        before = float(t_off["off_epa_prev"].iloc[0])
        after  = float(t_off["off_epa_last3"].iloc[0])
        eje_label = "EPA por jugada (ataque)"
        mejora_bool = (after - before) > 0
    else:
        before = float(t_def["def_epa_allowed_prev"].iloc[0])
        after  = float(t_def["def_epa_allowed_last3"].iloc[0])
        eje_label = "EPA por jugada permitido (defensa)"
        mejora_bool = (after - before) < 0  # mejora si permiten menos

    # ----- Gráfico HORIZONTAL: Previas vs Últimas 3 -----
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)

    y_pos  = np.array([0, 1])  # 0 = Previas, 1 = Últimas 3
    vals   = [before, after]
    labels = ["Previas", "Últimas 3"]
    color  = GREEN if mejora_bool else RED

    # margen horizontal extra para dar perspectiva y evitar choques con los bordes
    xmin, xmax = min(vals), max(vals)
    rng = max(xmax - xmin, 1e-6)
    pad_left, pad_right = 0.12*rng, 0.18*rng
    ax.set_xlim(xmin - pad_left, xmax + pad_right)

    # barras
    ax.barh(y_pos, vals, color=color, height=0.45, alpha=0.92, zorder=2)

    # valores con cajita semitransparente
    for y, v in zip(y_pos, vals):
        x = v + (0.012*rng if v >= 0 else -0.012*rng)
        ha = "left" if v >= 0 else "right"
        ax.text(x, y, f"{v:+.3f}", va="center", ha=ha, fontsize=12, color=FG,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#00000040", edgecolor="none"))

    # ejes y rejilla
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, color=FG)
    ax.set_xlabel(eje_label, fontsize=12, color=FG, labelpad=6)
    ax.grid(axis="x", linestyle="--", alpha=0.25, color=GRID, zorder=1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Título y subtítulo fuera del área del eje (evita solapes)
    fig.text(0.50, 0.96, f"Tendencia {team_choice} — {unidad_sel}",
             ha="center", va="top", color=FG, fontsize=16, fontweight="bold")
    fig.text(0.50, 0.91, f"ΔEPA = {after - before:+.3f}  (Últimas 3 vs Previas)",
             ha="center", va="top", color="#B9BDC7", fontsize=11)

    # Logo arriba-izquierda en coordenadas de figura
    im = load_logo_image(team_choice, base_zoom=0.11)
    if im is not None:
        ab = AnnotationBbox(im, (0.08, 0.92), frameon=False, xycoords=fig.transFigure, zorder=3)
        fig.add_artist(ab)

    # Fuente
    fig.text(0.01, 0.01,
             f"Fuente: nflverse-data  ·  NFL {SEASON}  ·  Jugadas pase+carrera",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")

    # Firma
    fig.text(0.99, 0.01, "@CuartayDato", ha="right", va="bottom",
             color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

    out_name = f"trending_{team_choice}_{'OFF' if unidad_sel=='Ataque' else 'DEF'}_{SEASON}.png"
    plt.savefig(out_name, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"Grafico generado: {out_name}")
    print(f"Unidad: {unidad_sel} | {'Mejora' if mejora_bool else 'Empeora'}")

if __name__ == "__main__":
    main()
