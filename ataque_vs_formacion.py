"""
ataque_vs_formacion.py
Eficiencia ofensiva de un equipo segun el paquete defensivo enfrentado.
Datos: PBP + pbp_participation (NGS) de nflverse.
  - Barras: EPA/jugada por tipo de cobertura (Cover 0, 1, 2, 3...)
  - Man vs Zona
  - Presion vs pocket limpio
  - Formacion (Base / Nickel / Dime)
NFL 2025
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = 2025
URL_PBP  = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
URL_PART = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{SEASON}.parquet"
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
DPI    = 200
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

MIN_PLAYS = 15   # jugadas minimas para mostrar valor

# Etiquetas mas legibles para coberturas
COV_LABELS = {
    "COVER_0": "Cover 0\n(Blitz total)",
    "COVER_1": "Cover 1\n(Man 1 alto)",
    "2_MAN":   "2-Man\n(Man 2 altos)",
    "COVER_2": "Cover 2\n(Zona 2 altos)",
    "COVER_3": "Cover 3\n(Zona 3 altos)",
    "COVER_4": "Cover 4\n(Quarters)",
    "COVER_6": "Cover 6\n(Mix Q/C2)",
    "COVER_9": "Cover 9\n(Quarters+)",
    "COMBO":   "Combo",
    "BLOWN":   "Blown\n(Error def.)",
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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


def count_dbs(s):
    """Cuenta DBs (CB+FS+SS+S) en cadena tipo '3 CB, 2 DT, 1 FS, 2 ILB, 2 OLB, 1 SS'."""
    if pd.isna(s):
        return None
    total = 0
    for pos in ["CB", "FS", "SS", "DB", "S"]:
        m = re.search(r"(\d+)\s+" + pos + r"(?:[,\s]|$)", str(s))
        if m:
            total += int(m.group(1))
    return total if total > 0 else None


def classify_formation(n):
    if n is None:
        return None
    if n <= 4:
        return "Base (<=4 DB)"
    if n == 5:
        return "Nickel (5 DB)"
    if n == 6:
        return "Dime (6 DB)"
    return "Dollar+ (7+)"


def bar_with_labels(ax, positions, values, colors, fontsize=8.5,
                    n_values=None, label_inside=True,
                    cats=None, league_avgs=None):
    """Dibuja barras horizontales con EPA + n dentro de la barra.
    Si se pasan cats y league_avgs, dibuja un marcador | blanco en la media de liga.
    """
    bars = ax.barh(positions, values, color=colors, height=0.65, zorder=3,
                   edgecolor="none")
    for i, (bar, val) in enumerate(zip(bars, values)):
        if np.isnan(val):
            continue
        y_mid = bar.get_y() + bar.get_height() / 2
        sign  = "+" if val >= 0 else ""
        n_str = f"n={int(n_values[i])}" if n_values is not None else ""

        # EPA y n apilados en el centro de la barra
        txt = f"{sign}{val:.3f}\n{n_str}" if n_str else f"{sign}{val:.3f}"
        ax.text(val / 2, y_mid, txt,
                va="center", ha="center",
                color="#0a0e13", fontsize=fontsize,
                fontweight="bold", zorder=5, linespacing=1.3)

        # Marcador de media de liga
        if cats is not None and league_avgs is not None:
            cat    = cats[i]
            lg_val = league_avgs.get(cat, None)
            if lg_val is not None:
                ax.scatter([lg_val], [y_mid], marker="|", s=250,
                           color="white", linewidths=2.5, zorder=6, alpha=0.9)
    return bars


def color_bars(values, norm):
    return [RYG(norm(v)) if not np.isnan(v) else GRID for v in values]


# ── INPUT ──────────────────────────────────────────────────────────────────────
team = input("Equipo ofensivo (siglas, p.ej. KC): ").strip().upper()

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
pbp = pd.read_csv(URL_PBP, low_memory=False, compression="infer")
to_num(pbp, ["epa", "wpa"])
print(f"PBP filas: {len(pbp):,}")

print(f"Descargando participacion {SEASON}...")
part = pd.read_parquet(URL_PART, columns=[
    "nflverse_game_id", "play_id",
    "defense_personnel", "defense_coverage_type", "defense_man_zone_type",
    "defenders_in_box", "number_of_pass_rushers", "was_pressure",
    "offense_formation",
])
print(f"Participacion filas: {len(part):,}")

# ── JOIN ───────────────────────────────────────────────────────────────────────
part = part.rename(columns={"nflverse_game_id": "game_id"})
pbp["play_id"]  = pd.to_numeric(pbp["play_id"],  errors="coerce")
part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")

merged = pbp.merge(part, on=["game_id", "play_id"], how="left")
print(f"Filas tras join: {len(merged):,}  |  "
      f"con cobertura: {merged['defense_coverage_type'].notna().sum():,}")

# ── FILTRAR JUGADAS OFENSIVAS DEL EQUIPO ───────────────────────────────────────
off = merged[
    (merged["posteam"] == team) &
    merged["play_type"].isin(["pass", "run"]) &
    merged["epa"].notna()
].copy()

if off.empty:
    raise SystemExit(f"No se encontraron jugadas ofensivas para {team}.")

print(f"\nJugadas ofensivas de {team}: {len(off):,}")

# ── DERIVAR COLUMNAS ──────────────────────────────────────────────────────────
off["db_count"]   = off["defense_personnel"].apply(count_dbs)
off["formation"]  = off["db_count"].apply(classify_formation)

# Normalizar man/zone
def man_zone(x):
    if pd.isna(x) or x == "":
        return None
    if "MAN" in str(x):
        return "Hombre"
    if "ZONE" in str(x):
        return "Zona"
    return None
off["man_zone"] = off["defense_man_zone_type"].apply(man_zone)

# Presion real (was_pressure si esta, fallback qb_hit|sack)
if off["was_pressure"].notna().sum() > 50:
    off["pressure"] = off["was_pressure"].map({True: "Bajo presion", False: "Pocket limpio"})
else:
    press_mask = (
        (pd.to_numeric(off.get("qb_hit", pd.Series(0)), errors="coerce").fillna(0) == 1) |
        (pd.to_numeric(off.get("sack",   pd.Series(0)), errors="coerce").fillna(0) == 1)
    )
    off["pressure"] = press_mask.map({True: "Bajo presion", False: "Pocket limpio"})

# ── MEDIA DE LIGA (todas las jugadas ofensivas, todos los equipos) ────────────
all_off = merged[
    merged["play_type"].isin(["pass", "run"]) &
    merged["epa"].notna()
].copy()
all_off["db_count"]  = all_off["defense_personnel"].apply(count_dbs)
all_off["formation"] = all_off["db_count"].apply(classify_formation)
all_off["man_zone"]  = all_off["defense_man_zone_type"].apply(man_zone)
if all_off["was_pressure"].notna().sum() > 50:
    all_off["pressure"] = all_off["was_pressure"].map({True: "Bajo presion", False: "Pocket limpio"})
else:
    press_mask_all = (
        (pd.to_numeric(all_off.get("qb_hit", pd.Series(0)), errors="coerce").fillna(0) == 1) |
        (pd.to_numeric(all_off.get("sack",   pd.Series(0)), errors="coerce").fillna(0) == 1)
    )
    all_off["pressure"] = press_mask_all.map({True: "Bajo presion", False: "Pocket limpio"})

lg_cov  = all_off.groupby("defense_coverage_type")["epa"].mean().to_dict()
lg_mz   = all_off.groupby("man_zone")["epa"].mean().to_dict()
lg_form = all_off.groupby("formation")["epa"].mean().to_dict()
lg_pres = all_off.groupby("pressure")["epa"].mean().to_dict()

# ── AGRUPACIONES ──────────────────────────────────────────────────────────────
def epa_group(df, col, order=None):
    """Devuelve DataFrame con EPA/play y n por categoria."""
    grp = (
        df[df[col].notna()]
        .groupby(col)["epa"]
        .agg(epa="mean", n="count")
        .reset_index()
        .rename(columns={col: "cat"})
    )
    grp = grp[grp["n"] >= MIN_PLAYS].copy()
    if order:
        grp["cat"] = pd.Categorical(grp["cat"], categories=order, ordered=True)
        grp = grp.sort_values("cat").dropna(subset=["cat"])
    else:
        grp = grp.sort_values("epa")
    return grp

cov_order  = ["COVER_0","COVER_1","2_MAN","COVER_2","COVER_3","COVER_4","COVER_6","COVER_9","COMBO","BLOWN"]
mz_order   = ["Zona", "Hombre"]
form_order = ["Base (<=4 DB)", "Nickel (5 DB)", "Dime (6 DB)", "Dollar+ (7+)"]
pres_order = ["Pocket limpio", "Bajo presion"]

cov_df  = epa_group(off, "defense_coverage_type")
cov_df["label"] = cov_df["cat"].apply(lambda x: COV_LABELS.get(x, x))
mz_df   = epa_group(off, "man_zone",   mz_order)
form_df = epa_group(off, "formation",  form_order)
pres_df = epa_group(off, "pressure",   pres_order)

# ── CONSOLA ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {team} vs paquetes defensivos | NFL {SEASON}")
print(f"{'='*60}")
print("\nCOBERTURA (EPA/jugada):")
for _, r in cov_df.iterrows():
    print(f"  {r['cat']:<12}  {r['epa']:+.3f}  ({int(r['n'])} jug.)")
print("\nHOMBRE vs ZONA:")
for _, r in mz_df.iterrows():
    print(f"  {r['cat']:<14}  {r['epa']:+.3f}  ({int(r['n'])} jug.)")
print()

# ── FIGURA ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.55, wspace=0.38,
    left=0.04, right=0.97,
    top=0.87,  bottom=0.06,
    width_ratios=[3, 1.4, 1.4],
)

ax_cov  = fig.add_subplot(gs[:, 0])   # izquierda: coberturas (2 filas)
ax_mz   = fig.add_subplot(gs[0, 1])   # arriba centro: man/zona
ax_pres = fig.add_subplot(gs[0, 2])   # arriba derecha: presion
ax_form = fig.add_subplot(gs[1, 1:])  # abajo centro+derecha: formacion

for ax in (ax_cov, ax_mz, ax_pres, ax_form):
    ax.set_facecolor("#151924")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=FG)

# Normalize comun para colores
all_epa = pd.concat([cov_df["epa"], mz_df["epa"], form_df["epa"], pres_df["epa"]]).dropna()
if len(all_epa):
    v_abs  = max(abs(all_epa.min()), abs(all_epa.max()), 0.05)
    norm   = Normalize(vmin=-v_abs, vmax=v_abs)
else:
    norm   = Normalize(vmin=-0.2, vmax=0.2)

# ── AX_COV: coberturas ────────────────────────────────────────────────────────
cov_sorted = cov_df.sort_values("epa").reset_index(drop=True)
positions  = np.arange(len(cov_sorted))
colors_cov = color_bars(cov_sorted["epa"].values, norm)

bar_with_labels(ax_cov, positions, cov_sorted["epa"].values, colors_cov,
                fontsize=8, n_values=cov_sorted["n"].values,
                cats=cov_sorted["cat"].values, league_avgs=lg_cov)
ax_cov.axvline(0, color=FG, linewidth=0.8, alpha=0.4, zorder=2)
ax_cov.set_yticks(positions)
ax_cov.set_yticklabels(cov_sorted["label"].tolist(), color=FG, fontsize=9)
ax_cov.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)
ax_cov.set_xlabel("EPA / jugada", color=FG, fontsize=9)
ax_cov.set_title("Por tipo de cobertura defensiva", color=FG, fontsize=10, pad=6, loc="left")
ax_cov.scatter([], [], marker="|", s=150, color="white", linewidths=2,
               label="| Media liga", alpha=0.9)
ax_cov.legend(fontsize=8, labelcolor=FG, facecolor="#151924",
              edgecolor=GRID, framealpha=0.5, loc="lower right")

# ── AX_MZ: man vs zona ────────────────────────────────────────────────────────
if not mz_df.empty:
    mz_vals   = mz_df["epa"].values
    mz_colors = color_bars(mz_vals, norm)
    mz_pos    = np.arange(len(mz_df))
    bar_with_labels(ax_mz, mz_pos, mz_vals, mz_colors,
                    n_values=mz_df["n"].values,
                    cats=mz_df["cat"].values, league_avgs=lg_mz)
    ax_mz.axvline(0, color=FG, linewidth=0.8, alpha=0.4, zorder=2)
    ax_mz.set_yticks(mz_pos)
    ax_mz.set_yticklabels(mz_df["cat"].tolist(), color=FG, fontsize=10)
ax_mz.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)
ax_mz.set_xlabel("EPA / jugada", color=FG, fontsize=9)
ax_mz.set_title("Hombre vs Zona", color=FG, fontsize=10, pad=6, loc="left")

# ── AX_PRES: presion ──────────────────────────────────────────────────────────
if not pres_df.empty:
    pr_vals   = pres_df["epa"].values
    pr_colors = color_bars(pr_vals, norm)
    pr_pos    = np.arange(len(pres_df))
    bar_with_labels(ax_pres, pr_pos, pr_vals, pr_colors,
                    n_values=pres_df["n"].values,
                    cats=pres_df["cat"].values, league_avgs=lg_pres)
    ax_pres.axvline(0, color=FG, linewidth=0.8, alpha=0.4, zorder=2)
    ax_pres.set_yticks(pr_pos)
    ax_pres.set_yticklabels(pres_df["cat"].tolist(), color=FG, fontsize=9.5)
ax_pres.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)
ax_pres.set_xlabel("EPA / jugada", color=FG, fontsize=9)
ax_pres.set_title("Presion vs Pocket limpio", color=FG, fontsize=10, pad=6, loc="left")

# ── AX_FORM: formacion ────────────────────────────────────────────────────────
if not form_df.empty:
    fm_vals   = form_df["epa"].values
    fm_colors = color_bars(fm_vals, norm)
    fm_pos    = np.arange(len(form_df))
    bar_with_labels(ax_form, fm_pos, fm_vals, fm_colors,
                    n_values=form_df["n"].values,
                    cats=form_df["cat"].values, league_avgs=lg_form)
    ax_form.axvline(0, color=FG, linewidth=0.8, alpha=0.4, zorder=2)
    ax_form.set_yticks(fm_pos)
    ax_form.set_yticklabels(form_df["cat"].tolist(), color=FG, fontsize=9)
ax_form.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)
ax_form.set_xlabel("EPA / jugada", color=FG, fontsize=9)
ax_form.set_title("Formacion defensiva (DBs en campo)", color=FG, fontsize=10, pad=6, loc="left")

# ── LOGO + TITULO ─────────────────────────────────────────────────────────────
logo = load_logo(team, base_zoom=0.10)
if logo is not None:
    logo_ax = fig.add_axes([0.03, 0.912, 0.065, 0.082])
    logo_ax.imshow(logo.get_data())
    logo_ax.axis("off")
    title_text = f"Eficiencia ofensiva vs paquetes defensivos | NFL {SEASON}"
else:
    title_text = f"{team} | Eficiencia ofensiva vs paquetes defensivos | NFL {SEASON}"

# ── TITULOS ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.975, title_text,
         ha="center", va="top", fontsize=15, fontweight="bold", color=FG)
fig.text(0.5, 0.928,
         "Datos: nflverse PBP + NGS participation  |  Cobertura y presion reales por jugada",
         ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  |  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

outfile = f"ataque_vs_formacion_{team}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
