"""
cuarto_down.py
Calidad de decisiones del HC en 4to down: ¿acierta cuando debe ir a por ello
y cuando debe patear? Modelo analítico propio basado en ydstogo + posición +
win probability (go_boost no disponible en nflverse PBP 2025).
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON       = 2025
DPI          = 200
BG           = "#0f1115"
FG           = "#EDEDED"
GRID         = "#2a2f3a"
RYG          = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
MIN_PLAYS    = 15   # mínimo de 4to-down decisiones para incluir al equipo

URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

# ── MODELO DE RECOMENDACIÓN ────────────────────────────────────────────────────
def recommend_go(ydstogo, yardline_100, wp_pos):
    """
    Modelo simplificado que aproxima el consenso analítico (4th Down Bot / EPA models).
    wp_pos: win probability de la posesión (0=perder casi seguro, 1=ganar casi seguro).
    Devuelve True (ir), False (patear), o None (dato inválido).
    """
    try:
        yds = int(ydstogo)
        yl  = float(yardline_100)   # 100 = propia línea de fondo, 1 = end zone rival
        wp  = float(wp_pos)
    except (TypeError, ValueError):
        return None

    # Desesperado: ir a por ello si la distancia es alcanzable
    if wp < 0.20 and yds <= 5:
        return True
    if wp < 0.10 and yds <= 8:
        return True

    # Cómodo líder: evitar arriesgar salvo distancia muy corta
    if wp > 0.88 and yds >= 3:
        return False

    # Umbrales analíticos estándar (campo + distancia)
    if yds <= 1:
        return yl <= 72       # casi siempre ir, salvo territorio propio profundo
    elif yds == 2:
        return yl <= 52       # ir en campo rival
    elif yds <= 4:
        return yl <= 42       # ir en territorio rival
    elif yds == 5:
        return yl <= 33       # ir en zona roja
    else:
        return False          # distancia larga → patear

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.038):
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

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
print(f"Filas: {len(df):,}")

# ── FILTRO 4TO DOWN ────────────────────────────────────────────────────────────
mask = (
    (df["down"] == 4) &
    df["posteam"].notna() &
    df["ydstogo"].notna() &
    df["yardline_100"].notna() &
    df["play_type"].isin(["pass", "run", "punt", "field_goal"])
)
df4 = df[mask].copy()
print(f"Jugadas 4to down válidas: {len(df4):,}")

# ── WIN PROBABILITY DE LA POSESIÓN ────────────────────────────────────────────
# wp en nflverse es del equipo local; calcular WP del equipo en posesión
if "home_team" in df4.columns and "home_wp" in df4.columns and "away_wp" in df4.columns:
    df4["wp_pos"] = np.where(
        df4["posteam"] == df4["home_team"],
        pd.to_numeric(df4["home_wp"], errors="coerce"),
        pd.to_numeric(df4["away_wp"], errors="coerce"),
    )
elif "wp" in df4.columns:
    df4["wp_pos"] = pd.to_numeric(df4["wp"], errors="coerce")
else:
    df4["wp_pos"] = 0.5   # fallback neutro

df4["wp_pos"] = df4["wp_pos"].fillna(0.5)

# ── RECOMENDACIÓN Y DECISIÓN ───────────────────────────────────────────────────
df4["rec_go"] = df4.apply(
    lambda r: recommend_go(r["ydstogo"], r["yardline_100"], r["wp_pos"]), axis=1
)
df4 = df4[df4["rec_go"].notna()].copy()
df4["rec_go"]    = df4["rec_go"].astype(bool)
df4["went"]      = df4["play_type"].isin(["pass", "run"])
df4["correct"]   = df4["rec_go"] == df4["went"]
print(f"Jugadas clasificadas: {len(df4):,}")

# ── AGREGAR POR EQUIPO ─────────────────────────────────────────────────────────
records = []
for team, g in df4.groupby("posteam"):
    go_pl   = g[g["rec_go"]]
    kick_pl = g[~g["rec_go"]]
    records.append({
        "posteam":    team,
        "n_plays":    len(g),
        "accuracy":   g["correct"].mean() * 100,
        "acc_go":     go_pl["correct"].mean()   * 100 if len(go_pl)   > 0 else np.nan,
        "acc_kick":   kick_pl["correct"].mean() * 100 if len(kick_pl) > 0 else np.nan,
        "n_go_rec":   len(go_pl),
        "n_kick_rec": len(kick_pl),
    })

stats = pd.DataFrame(records)
stats = (
    stats[stats["n_plays"] >= MIN_PLAYS]
       .sort_values("accuracy", ascending=True)
       .reset_index(drop=True)
)
n_teams = len(stats)
print(f"Equipos incluidos: {n_teams}")

avg_acc  = stats["accuracy"].mean()
avg_go   = stats["acc_go"].mean()
avg_kick = stats["acc_kick"].mean()

# ── CONSOLA ────────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  Decisiones correctas en 4to down | NFL {SEASON}")
print(f"  Modelo: ydstogo + yardline_100 + win probability")
print(f"{'='*68}")
print(f"{'Equipo':<6} {'Total%':>8} {'n':>5} {'Go%':>8} {'nGo':>5} {'Kick%':>8} {'nKick':>6}")
print("-" * 57)
for _, r in stats.sort_values("accuracy", ascending=False).iterrows():
    go_s   = f"{r['acc_go']:>7.1f}%"   if not np.isnan(r["acc_go"])   else f"{'—':>8}"
    kick_s = f"{r['acc_kick']:>7.1f}%" if not np.isnan(r["acc_kick"]) else f"{'—':>8}"
    print(f"{r['posteam']:<6} {r['accuracy']:>7.1f}% {int(r['n_plays']):>5}"
          f" {go_s} {int(r['n_go_rec']):>5} {kick_s} {int(r['n_kick_rec']):>6}")
print(f"\nLiga avg: {avg_acc:.1f}% | Go: {avg_go:.1f}% | Kick: {avg_kick:.1f}%\n")

# ── FIGURA ─────────────────────────────────────────────────────────────────────
BAR_H  = 0.58
LOGO_X = -13.0
XLEFT  = -17.5
XRIGHT = 132.0

fig_w = 13.5
fig_h = max(10, n_teams * 0.56 + 3.2)

fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(XLEFT, XRIGHT)
ax.set_ylim(-0.85, n_teams + 1.0)

norm = Normalize(
    vmin=max(35, stats["accuracy"].min() - 3),
    vmax=min(100, stats["accuracy"].max() + 3),
)

# ── FONDOS ALTERNADOS ─────────────────────────────────────────────────────────
for i in range(n_teams):
    bg_c = "#161b25" if i % 2 == 0 else BG
    ax.add_patch(plt.Rectangle(
        (XLEFT, i - BAR_H / 2 - 0.07), XRIGHT - XLEFT, BAR_H + 0.14,
        color=bg_c, zorder=0,
    ))

# ── LÍNEA PROMEDIO LIGA ───────────────────────────────────────────────────────
ax.axvline(avg_acc, color="#888888", linewidth=1.1, linestyle="--", alpha=0.65, zorder=1)
ax.text(avg_acc, n_teams + 0.12, f"Liga\n{avg_acc:.1f}%",
        ha="center", va="bottom", color="#aaaaaa", fontsize=7.5, linespacing=1.3)

# ── BARRAS, LOGOS Y ETIQUETAS ─────────────────────────────────────────────────
for i, row in stats.iterrows():
    team     = row["posteam"]
    acc      = row["accuracy"]
    acc_go   = row["acc_go"]
    acc_kick = row["acc_kick"]

    color = RYG(norm(acc))
    ax.barh(i, acc, height=BAR_H, color=color, alpha=0.88, zorder=2, left=0)

    logo = load_logo(team, base_zoom=0.033)
    if logo is not None:
        ab = AnnotationBbox(logo, (LOGO_X, i),
                            frameon=False, zorder=4,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.text(LOGO_X, i, team, ha="center", va="center",
                color=FG, fontsize=8, fontweight="bold")

    ax.text(acc + 1.0, i, f"{acc:.1f}%",
            ha="left", va="center", color=FG,
            fontsize=8.5, fontweight="bold", zorder=3)

    go_str   = f"↑ {acc_go:.0f}%"   if not np.isnan(acc_go)   else "↑ —"
    kick_str = f"↓ {acc_kick:.0f}%" if not np.isnan(acc_kick) else "↓ —"
    ax.text(105, i + 0.145, go_str,
            ha="left", va="center",
            color="#06d6a0", fontsize=7.5, fontweight="bold", zorder=3)
    ax.text(105, i - 0.145, kick_str,
            ha="left", va="center",
            color="#ffd166", fontsize=7.5, fontweight="bold", zorder=3)

ax.text(105, n_teams + 0.12, "Acierta al ir ↑ / Acierta al patear ↓",
        ha="left", va="bottom", color="#aaaaaa", fontsize=7.5, fontweight="bold")

# ── EJES ──────────────────────────────────────────────────────────────────────
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("% de decisiones correctas", color=FG, fontsize=9.5)
ax.tick_params(colors=FG, labelsize=8)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.axvline(0, color=GRID, linewidth=0.8, zorder=1)
ax.xaxis.grid(color=GRID, linewidth=0.4, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), color=FG, fontsize=8)

# ── TÍTULOS ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.99,
         f"HC acertando en 4to down | NFL {SEASON}",
         ha="center", va="top", color=FG, fontsize=14, fontweight="bold")
fig.text(0.5, 0.976,
         "% de veces que la decisión real coincide con el análisis situacional  ·  "
         "Verde ↑ = acierta al ir a por ello  ·  Amarillo ↓ = acierta al patear",
         ha="center", va="top", color="#888888", fontsize=8.5, fontstyle="italic")
fig.text(0.01, 0.005,
         f"Fuente: nflverse PBP {SEASON}  |  Modelo propio: ydstogo + posición + win probability",
         ha="left", va="bottom", color="#555555", fontsize=7.5, fontstyle="italic")
fig.text(0.99, 0.005, "@CuartayDato",
         ha="right", va="bottom", color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

out = f"cuarto_down_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
