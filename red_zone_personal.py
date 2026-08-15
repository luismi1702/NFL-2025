"""
red_zone_personal.py
Red Zone: EPA según personal ofensivo.
  · Equipo específico → barras ataque + defensa con rankings y comparativa campo abierto
  · Enter           → heatmap grid 32 equipos (ataque + defensa)
NFL 2025
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, cargar_participation, DatosNoDisponibles, salida, season_cli, sello
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SEASON  = season_cli()   # None = auto-detectar última temporada

BG    = "#0f1115"
CARD  = "#151924"
CARD2 = "#1a2030"
FG    = "#EDEDED"
GRID  = "#2a2f3a"
DPI   = 170
LOGOS_DIR    = "logos"

MIN_SNAPS    = 10
MIN_SNAPS_TM = 8    # mínimo para modo equipo (menos jugadas en RZ)

RYG   = LinearSegmentedColormap.from_list("ryg",  ["#c0392b", "#e8b84b", "#27ae60"])
RYG_r = RYG.reversed()

OFF_PKG_ORDER = ["11", "12", "21", "13", "22", "10"]
OFF_PKG_LABEL = {
    "11": "11 personal", "12": "12 personal",
    "21": "21 personal", "13": "13 personal",
    "22": "22 personal", "10": "10 personal",
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_logo(team, zoom=0.035):
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
        z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * (zoom):
            z = 900.0 * (zoom) / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


def parse_off_pkg(s):
    if pd.isna(s): return None
    s = str(s)
    rb = re.search(r"(\d+)\s*RB", s, re.I)
    fb = re.search(r"(\d+)\s*FB", s, re.I)
    te = re.search(r"(\d+)\s*TE", s, re.I)
    rb_n = (int(rb.group(1)) if rb else 0) + (int(fb.group(1)) if fb else 0)
    te_n = int(te.group(1)) if te else 0
    return f"{rb_n}{te_n}" if rb_n > 0 else None


def get_ranks(df, team_col, pkg_order, ascending=False):
    """Devuelve {team: {pkg: rank}} para todos los equipos."""
    grp = (df[df["off_pkg"].notna()]
           .groupby([team_col, "off_pkg"])["epa"]
           .agg(epa="mean", n="count").reset_index())
    grp = grp[grp["n"] >= MIN_SNAPS_TM]
    ranks = {}
    for pkg in pkg_order:
        sub = grp[grp["off_pkg"] == pkg].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("epa", ascending=ascending).reset_index(drop=True)
        for i, row in sub.iterrows():
            ranks.setdefault(row[team_col], {})[pkg] = int(sub.index[sub[team_col] == row[team_col]][0] + 1)
    return ranks


# ── INPUT ─────────────────────────────────────────────────────────────────────
team = input("Equipo (siglas, ej: SF) o Enter para grid 32 equipos: ").strip().upper()
modo = "equipo" if team else "grid"

# ── CARGA ─────────────────────────────────────────────────────────────────────
pbp, SEASON = cargar_pbp(SEASON)
pbp["epa"]     = pd.to_numeric(pbp["epa"],     errors="coerce")
pbp["play_id"] = pd.to_numeric(pbp["play_id"], errors="coerce")
print(f"PBP {SEASON}: {len(pbp):,} jugadas REG")

try:
    part, _ = cargar_participation(SEASON)
except DatosNoDisponibles as e:
    raise SystemExit(
        "\n  No se puede generar este grafico todavia.\n"
        f"  {e}\n"
        "  Este visual necesita datos de participacion (el personal en campo en zona roja),\n"
        "  que nflverse publica mas tarde que el play-by-play.\n")
part = part[["nflverse_game_id", "play_id", "offense_personnel"]]
part = part.rename(columns={"nflverse_game_id": "game_id"})
part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")

merged    = pbp.merge(part, on=["game_id", "play_id"], how="left")
all_plays = merged[merged["play_type"].isin(["pass", "run"]) & merged["epa"].notna()].copy()
all_plays["off_pkg"] = all_plays["offense_personnel"].apply(parse_off_pkg)

rz_plays = all_plays[all_plays["yardline_100"].le(20)].copy()
op_plays  = all_plays[~all_plays["yardline_100"].le(20)].copy()   # open field (no RZ)
print(f"Jugadas totales: {len(all_plays):,}  |  Red zone: {len(rz_plays):,}")

if modo == "equipo" and team not in rz_plays["posteam"].values:
    raise SystemExit(f"No se encontraron jugadas en red zone para {team}.")


# ══════════════════════════════════════════════════════════════════════════════
# MODO EQUIPO — barras con ranking y comparativa campo abierto
# ══════════════════════════════════════════════════════════════════════════════
if modo == "equipo":

    # Medias de liga en RZ por paquete
    lg_rz = (rz_plays[rz_plays["off_pkg"].notna()]
             .groupby("off_pkg")["epa"].mean().to_dict())
    lg_op  = (op_plays[op_plays["off_pkg"].notna()]
              .groupby("off_pkg")["epa"].mean().to_dict())

    # Rankings de ataque (RZ)
    rk_off = get_ranks(rz_plays, "posteam", OFF_PKG_ORDER, ascending=False).get(team, {})
    # Rankings de defensa (RZ) — ascending=True porque menor EPA = mejor defensa
    rk_def = get_ranks(rz_plays, "defteam", OFF_PKG_ORDER, ascending=True).get(team, {})

    def pkg_stats(df, team_col, team_val, min_s=MIN_SNAPS_TM):
        sub = df[df[team_col] == team_val]
        result = {}
        for pkg in OFF_PKG_ORDER:
            s = sub[sub["off_pkg"] == pkg].dropna(subset=["epa"])
            if len(s) >= min_s:
                result[pkg] = (s["epa"].mean(), len(s))
        return result

    off_stats = pkg_stats(rz_plays, "posteam", team)
    def_stats  = pkg_stats(rz_plays, "defteam", team)
    off_op     = pkg_stats(op_plays,  "posteam", team, min_s=20)  # campo abierto

    # ── FIGURA ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                             hspace=0.3, wspace=0.42,
                             left=0.06, right=0.97,
                             top=0.87, bottom=0.06)

    ax_off = fig.add_subplot(gs[0])
    ax_def = fig.add_subplot(gs[1])

    def draw_bar_panel(ax, stats, ranks, lg_rz_dict, op_stats, cmap, title, xlabel):
        pkgs   = [p for p in OFF_PKG_ORDER if p in stats]
        if not pkgs:
            ax.axis("off")
            return

        epa_vals  = [stats[p][0] for p in pkgs]
        n_vals    = [stats[p][1] for p in pkgs]
        labels    = [OFF_PKG_LABEL[p] for p in pkgs]
        positions = np.arange(len(pkgs))

        # Norma vs media de liga RZ
        lg_vals  = [lg_rz_dict.get(p, 0) for p in pkgs]
        devs     = [e - l for e, l in zip(epa_vals, lg_vals)]
        d_abs    = max(abs(min(devs)), abs(max(devs)), 0.03) if devs else 0.2
        norm_obj = Normalize(vmin=-d_abs, vmax=d_abs)
        colors   = [cmap(norm_obj(d)) for d in devs]

        ax.barh(positions, epa_vals, color=colors, height=0.6, zorder=3, edgecolor="none")

        # Marcador media de liga RZ
        for i, pkg in enumerate(pkgs):
            lg = lg_rz_dict.get(pkg)
            if lg is not None:
                ax.scatter([lg], [positions[i]], marker="|", s=250,
                           color="white", linewidths=2, zorder=5, alpha=0.9)

        # Marcador campo abierto (diamante)
        for i, pkg in enumerate(pkgs):
            if pkg in op_stats:
                ax.scatter([op_stats[pkg][0]], [positions[i]], marker="D", s=40,
                           color="#ffd166", linewidths=0, zorder=6, alpha=0.85)

        # Etiquetas de valor + rank
        max_bar = max(abs(v) for v in epa_vals) if epa_vals else 0.1
        for i, (val, n, pkg) in enumerate(zip(epa_vals, n_vals, pkgs)):
            sign  = "+" if val >= 0 else ""
            rank  = ranks.get(pkg)
            r_str = f"  #{rank}" if rank else ""
            txt   = f"{sign}{val:.3f}  (n={n}){r_str}"
            inside = abs(val) >= max_bar * 0.3
            if inside:
                ax.text(val / 2, positions[i], txt,
                        ha="center", va="center",
                        color="#0a0e13", fontsize=7.5, fontweight="bold", zorder=6)
            else:
                offset = max_bar * 0.04
                ax.text(val + (offset if val >= 0 else -offset), positions[i], txt,
                        ha="left" if val >= 0 else "right", va="center",
                        color=FG, fontsize=7.5, fontweight="bold", zorder=6)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, color=FG, fontsize=9)
        ax.axvline(0, color=FG, linewidth=0.7, alpha=0.4, zorder=2)
        ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.4, zorder=1)
        ax.set_xlabel(xlabel, color=FG, fontsize=8.5)
        ax.set_title(title, color=FG, fontsize=10, pad=6, fontweight="bold", loc="left")
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=FG, length=0)

        # Leyendas
        ax.scatter([], [], marker="|", s=150, color="white", linewidths=2,
                   label="| Media liga (RZ)")
        ax.scatter([], [], marker="D", s=35, color="#ffd166", linewidths=0,
                   label="◆ EPA campo abierto")
        ax.legend(fontsize=7, labelcolor=FG, facecolor=CARD,
                  edgecolor=GRID, framealpha=0.6, loc="lower right")

    draw_bar_panel(ax_off, off_stats, rk_off, lg_rz, off_op,
                   RYG,
                   f"Red Zone — Ataque  {team}",
                   "EPA / jugada  (verde = mejor que media liga RZ)")

    draw_bar_panel(ax_def, def_stats, rk_def, lg_rz, {},
                   RYG_r,
                   f"Red Zone — Defensa  {team}",
                   "EPA permitido  (verde = mejor que media liga RZ)")

    # Logo
    logo = load_logo(team, zoom=0.095)
    if logo:
        lax = fig.add_axes([0.03, 0.912, 0.055, 0.075])
        lax.imshow(logo.get_data())
        lax.axis("off")

    fig.text(0.5, 0.975,
             f"{team}  |  Red Zone por Personal Ofensivo  |  NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.952,
             "| = media de liga en RZ  ·  ◆ = EPA del mismo equipo en campo abierto  ·  #N = ranking liga",
             ha="center", va="top", fontsize=8, color="#888", fontstyle="italic")
    fig.text(0.01, 0.008,
             f"Fuente: nflverse PBP + NGS participation  |  {sello(SEASON)}  |  Mín {MIN_SNAPS_TM} snaps en RZ",
             ha="left", va="bottom", fontsize=7, color="#555", fontstyle="italic")
    fig.text(0.99, 0.008, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888", alpha=0.8, fontstyle="italic")

    outfile = salida(f"red_zone_personal_{team}_{SEASON}.png", SEASON)
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {outfile}")


# ══════════════════════════════════════════════════════════════════════════════
# MODO GRID — heatmap 32 equipos
# ══════════════════════════════════════════════════════════════════════════════
else:

    def build_pivot(df, team_col):
        grp = (df[df["off_pkg"].notna()]
               .groupby([team_col, "off_pkg"])["epa"]
               .agg(epa="mean", n="count").reset_index())
        grp = grp[grp["n"] >= MIN_SNAPS]
        pivot = grp.pivot(index=team_col, columns="off_pkg", values="epa")
        # Solo paquetes con algun dato (elimina columnas 100% vacias como "10")
        cols = [c for c in OFF_PKG_ORDER if c in pivot.columns and pivot[c].notna().any()]
        return pivot.reindex(columns=cols)

    def build_pivot_n(df, team_col):
        grp = (df[df["off_pkg"].notna()]
               .groupby([team_col, "off_pkg"])["epa"]
               .count().reset_index())
        grp.columns = [team_col, "off_pkg", "n"]
        grp = grp[grp["n"] >= MIN_SNAPS]
        pivot = grp.pivot(index=team_col, columns="off_pkg", values="n")
        cols = [c for c in OFF_PKG_ORDER if c in pivot.columns and pivot[c].notna().any()]
        return pivot.reindex(columns=cols)

    off_rz = build_pivot(rz_plays, "posteam")
    def_rz = build_pivot(rz_plays, "defteam")

    # ascending=True: peor ataque en row_i=0 (fondo), mejor ataque en row_i=n-1 (arriba)
    off_rz_total = rz_plays.groupby("posteam")["epa"].mean().reindex(off_rz.index).sort_values(ascending=True)
    off_rz = off_rz.reindex(off_rz_total.index)

    # ascending=False: peor defensa (EPA alto) en row_i=0 (fondo), mejor defensa (EPA bajo) arriba
    def_rz_total = rz_plays.groupby("defteam")["epa"].mean().reindex(def_rz.index).sort_values(ascending=False)
    def_rz = def_rz.reindex(def_rz_total.index)

    off_rz_n = build_pivot_n(rz_plays, "posteam").reindex(off_rz.index)
    def_rz_n = build_pivot_n(rz_plays, "defteam").reindex(def_rz.index)

    def draw_hm(ax, pivot, pivot_n, cmap, title):
        data  = pivot.values.astype(float)
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            ax.axis("off")
            return
        # Percentiles 5-95 en vez de min/max: un outlier con n minimo
        # (ej. -1.95 con n=10) no debe aplanar la escala del resto
        v_abs = max(abs(np.nanpercentile(data, 5)),
                    abs(np.nanpercentile(data, 95)), 0.05)
        norm  = Normalize(vmin=-v_abs, vmax=v_abs, clip=True)
        n_teams, n_pkgs = len(pivot.index), len(pivot.columns)

        # Alternating row backgrounds for readability
        for row_i in range(n_teams):
            row_bg = "#181e2a" if row_i % 2 == 0 else "#121620"
            ax.add_patch(plt.Rectangle((-1.3, row_i - 0.5), 1.3 + n_pkgs, 1,
                                        color=row_bg, zorder=0))

        for row_i in range(n_teams):
            for col_j in range(n_pkgs):
                val = data[row_i, col_j]
                bg  = cmap(norm(val)) if not np.isnan(val) else "#1a2030"
                ax.add_patch(plt.Rectangle((col_j - 0.5, row_i - 0.5), 1, 1,
                                            color=bg, zorder=1))
                if not np.isnan(val):
                    n_val = pivot_n.iloc[row_i, col_j] if pivot_n is not None else 0
                    n_val = int(n_val) if not (isinstance(n_val, float) and np.isnan(n_val)) else 0
                    sign  = "+" if val >= 0 else ""
                    lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2] if isinstance(bg, tuple) else 0
                    txt_col = "#0a0e13" if lum > 0.45 else FG
                    ax.text(col_j, row_i, f"{sign}{val:.2f}\n({n_val})",
                            ha="center", va="center",
                            color=txt_col, fontsize=7.5, fontweight="bold",
                            zorder=2, linespacing=1.2)

        # Subtle row separators
        for row_i in range(n_teams + 1):
            ax.axhline(row_i - 0.5, color="#0a0d13", linewidth=0.4, zorder=3)

        for row_i, tm in enumerate(pivot.index):
            logo = load_logo(str(tm), zoom=0.032)
            if logo:
                ab = AnnotationBbox(logo, (-0.9, row_i), xycoords="data",
                                    frameon=False, zorder=4, pad=0)
                ax.add_artist(ab)

        ax.set_xlim(-1.3, n_pkgs - 0.5)
        ax.set_ylim(-0.5, n_teams - 0.5)
        # Etiquetas de personal dibujadas manualmente encima del grid
        for col_j, pkg in enumerate(pivot.columns):
            ax.text(col_j, n_teams - 0.5 + 0.25,
                    OFF_PKG_LABEL.get(pkg, pkg),
                    ha="center", va="bottom",
                    color=FG, fontsize=8.5, fontweight="bold",
                    clip_on=False, zorder=10)
        ax.set_xticks([])
        ax.set_yticks(range(n_teams))
        ax.set_yticklabels([""] * n_teams)
        ax.tick_params(which="both", length=0, labelbottom=False, labeltop=False)
        ax.set_title(title, color=FG, fontsize=9.5, pad=38, fontweight="bold", loc="left")
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    # Dos PNGs separados (ataque / defensa): un solo lienzo de 32×2 paneles
    # hacía los textos de celda diminutos e ilegibles en móvil.
    def render_uno(pivot, pivot_n, cmap, titulo_seccion, lado, sufijo):
        fig = plt.figure(figsize=(17, 15.5), facecolor=BG)
        gs  = gridspec.GridSpec(1, 1, figure=fig,
                                 left=0.08, right=0.97, top=0.885, bottom=0.035)
        draw_hm(fig.add_subplot(gs[0]), pivot, pivot_n, cmap, titulo_seccion)
        fig.text(0.5, 0.968,
                 f"Red Zone por Personal — {lado}  |  Jugadas dentro de las 20 yardas  |  NFL {SEASON}",
                 ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
        fig.text(0.5, 0.944, f"Mínimo {MIN_SNAPS} snaps por celda  |  Valor: EPA medio  (n = snaps)",
                 ha="center", va="top", fontsize=9, color="#888", fontstyle="italic")
        fig.text(0.01, 0.006, f"Fuente: nflverse PBP + NGS participation  |  {sello(SEASON)}",
                 ha="left", va="bottom", fontsize=7.5, color="#555", fontstyle="italic")
        fig.text(0.99, 0.006, "@CuartayDato",
                 ha="right", va="bottom", fontsize=10, color="#888", alpha=0.8, fontstyle="italic")
        outfile = salida(f"red_zone_personal_{sufijo}_{SEASON}.png", SEASON)
        plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"Guardado: {outfile}")

    render_uno(off_rz, off_rz_n, RYG,
               "EPA ofensivo por personal  ·  mejor ataque arriba  (verde = mejor que media, rojo = peor)",
               "Ataque", "ataque")
    render_uno(def_rz, def_rz_n, RYG_r,
               "EPA permitido por personal rival  ·  mejor defensa arriba  (verde = buena defensa, rojo = vulnerable)",
               "Defensa", "defensa")
