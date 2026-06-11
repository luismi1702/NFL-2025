"""
matchup_intel.py
Intel táctica: A atacando vs B defendiendo.
Detecta mismatches automáticos (EXPLOIT / NEUTRO / RIESGO).
NFL 2025
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, cargar_participation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SEASON   = None   # None = auto-detectar última temporada

BG    = "#0f1115"
CARD  = "#151924"
CARD2 = "#1a2030"
FG    = "#EDEDED"
GRID  = "#2a2f3a"
DPI   = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

COL_EXPLOIT  = "#06d6a0"
COL_RISK     = "#d84a4a"
COL_NEUTRAL  = "#5a6070"
BG_EXPLOIT   = "#071f14"
BG_RISK      = "#1f0707"
BG_NEUTRAL   = "#12161e"

MIN_SNAPS           = 15
MISMATCH_THRESHOLD  = 0.06

RYG   = LinearSegmentedColormap.from_list("ryg", ["#c0392b", "#e8b84b", "#27ae60"])
RYG_r = RYG.reversed()

OFF_PKG_ORDER = ["11", "12", "21", "10", "13", "22"]
OFF_PKG_LABEL = {
    "11": "11 personal",
    "12": "12 personal",
    "21": "21 personal",
    "10": "10 personal",
    "13": "13 personal",
    "22": "22 personal",
}
COV_ORDER = ["COVER_0","COVER_1","2_MAN","COVER_2","COVER_3","COVER_4","COVER_6","COVER_9"]
COV_LABELS = {
    "COVER_0": "Cover 0",  "COVER_1": "Cover 1",  "2_MAN": "2-Man",
    "COVER_2": "Cover 2",  "COVER_3": "Cover 3",  "COVER_4": "Cover 4",
    "COVER_6": "Cover 6",  "COVER_9": "Cover 9",
}
MZ_ORDER   = ["Zona", "Hombre"]
PRES_ORDER = ["Pocket limpio", "Bajo presion"]


# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.07):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        zoom = base_zoom / HARD_PENALTY[team] if team in HARD_PENALTY else \
               base_zoom / np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def count_dbs(s):
    if pd.isna(s): return None
    total = sum(int(m.group(1))
                for pos in ["CB", "FS", "SS", "DB"]
                for m in [re.search(r"(\d+)\s+" + pos + r"(?:[,\s]|$)", str(s))]
                if m)
    return total if total > 0 else None


def classify_def_pkg(n):
    if n is None: return None
    if n <= 4: return "Base"
    if n == 5: return "Nickel"
    if n == 6: return "Dime"
    return "Dollar+"


def parse_off_pkg(s):
    if pd.isna(s): return None
    s = str(s)
    rb = re.search(r"(\d+)\s*RB", s, re.I)
    fb = re.search(r"(\d+)\s*FB", s, re.I)
    te = re.search(r"(\d+)\s*TE", s, re.I)
    rb_n = (int(rb.group(1)) if rb else 0) + (int(fb.group(1)) if fb else 0)
    te_n = int(te.group(1)) if te else 0
    return f"{rb_n}{te_n}" if rb_n > 0 else None


def man_zone_label(x):
    if pd.isna(x) or x == "": return None
    return "Hombre" if "MAN" in str(x) else ("Zona" if "ZONE" in str(x) else None)


def pressure_label(x):
    if pd.isna(x): return None
    return "Bajo presion" if int(float(x)) else "Pocket limpio"


def team_epa_by_cat(df, col, cats):
    result = {}
    for cat in cats:
        sub = df[df[col] == cat].dropna(subset=["epa"])
        if len(sub) >= MIN_SNAPS:
            result[cat] = (sub["epa"].mean(), len(sub))
        else:
            result[cat] = (np.nan, len(sub))
    return result


def league_epa_by_cat(all_plays, col, cats):
    grp = all_plays.groupby(col)["epa"].mean()
    return {cat: grp.get(cat, np.nan) for cat in cats}


def league_ranks(all_plays, team_col, col, cats, ascending=False):
    df  = all_plays[all_plays[col].notna()]
    grp = df.groupby([team_col, col])["epa"].agg(epa="mean", n="count").reset_index()
    grp = grp[grp["n"] >= MIN_SNAPS]
    ranks = {}
    for cat in cats:
        sub = grp[grp[col] == cat].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("epa", ascending=ascending).reset_index(drop=True)
        sub["rank"] = sub.index + 1
        for _, row in sub.iterrows():
            ranks.setdefault(row[team_col], {})[cat] = int(row["rank"])
    return ranks


def classify_mismatch(att_epa, def_epa, lg_epa):
    if np.isnan(att_epa) or np.isnan(def_epa) or np.isnan(lg_epa):
        return "NEUTRO", 0.0
    score = (att_epa - lg_epa) + (def_epa - lg_epa)
    if score >= MISMATCH_THRESHOLD:
        return "EXPLOIT", score
    if score <= -MISMATCH_THRESHOLD:
        return "RIESGO", score
    return "NEUTRO", score


# ── DRAW SECTION ──────────────────────────────────────────────────────────────
def draw_section(ax, rows, team_atk, team_def, rk_atk, rk_def, section_title):
    """
    Un solo axes por sección.
    Las barras nacen en ±CENTER_HALF y crecen hacia afuera.
    El centro [-CENTER_HALF, +CENTER_HALF] queda libre para etiquetas.
    """
    n = len(rows)
    if n == 0:
        ax.axis("off")
        return

    ROW_H   = 1.0
    TITLE_H = 0.55
    total_h = n * ROW_H + TITLE_H

    all_vals = [v for r in rows for v in [r["att_epa"], r["def_epa"]] if not np.isnan(v)]
    v_abs = max(abs(min(all_vals)), abs(max(all_vals)), 0.08) if all_vals else 0.3
    v_lim = v_abs * 2.2          # espacio total de cada lado
    CENTER_HALF = v_lim * 0.32   # zona central reservada — barras NO entran aquí
    norm = Normalize(vmin=-v_abs, vmax=v_abs)

    ax.set_xlim(-v_lim, v_lim)
    ax.set_ylim(-0.5, total_h - 0.5)
    ax.axis("off")

    # ── Cabecera de sección ───────────────────────────────────────────────────
    title_y = n * ROW_H - 0.5 + TITLE_H / 2
    ax.add_patch(plt.Rectangle((-v_lim, n * ROW_H - 0.5), v_lim * 2, TITLE_H,
                                color="#1e2535", zorder=1))
    ax.text(0, title_y, section_title,
            ha="center", va="center", color=FG,
            fontsize=10, fontweight="bold", zorder=2)
    ax.text(-v_lim * 0.7, title_y, f"◀  {team_atk}  EPA ofensivo",
            ha="center", va="center", color="#777", fontsize=7, zorder=2)
    ax.text( v_lim * 0.7, title_y, f"EPA cedido  {team_def}  ▶",
            ha="center", va="center", color="#777", fontsize=7, zorder=2)

    for i, r in enumerate(rows):
        y = n - i - 1
        mm_label, _ = classify_mismatch(r["att_epa"], r["def_epa"], r["lg_epa"])

        bg = BG_EXPLOIT if mm_label == "EXPLOIT" else (BG_RISK if mm_label == "RIESGO" else BG_NEUTRAL)
        ax.add_patch(plt.Rectangle((-v_lim, y - 0.5), v_lim * 2, ROW_H,
                                    color=bg, zorder=1))
        if i < n - 1:
            ax.axhline(y - 0.5, color=GRID, linewidth=0.4, alpha=0.25, zorder=2)

        # ── Barra atacante: nace en x=-CENTER_HALF y crece hacia la izquierda ─
        att = r["att_epa"]
        if not np.isnan(att):
            bar_len = abs(np.clip(att, -v_abs, v_abs))
            c = RYG(norm(att))
            ax.barh(y, bar_len, height=0.52, left=-(CENTER_HALF + bar_len),
                    color=c, zorder=3, edgecolor="none", alpha=0.95)
            rank  = rk_atk.get(r["cat_key"])
            sign  = "+" if att >= 0 else ""
            r_str = f"  #{rank}" if rank else ""
            ax.text(-(CENTER_HALF + bar_len) - v_lim * 0.04, y,
                    f"{sign}{att:.3f}{r_str}",
                    ha="right", va="center",
                    color=FG, fontsize=8.5, fontweight="bold", zorder=5)
        else:
            ax.text(-(CENTER_HALF + (v_lim - CENTER_HALF) * 0.5), y,
                    "n/d", ha="center", va="center", color="#555", fontsize=8)

        # ── Barra defensor: nace en x=+CENTER_HALF y crece hacia la derecha ──
        deff = r["def_epa"]
        if not np.isnan(deff):
            bar_len = abs(np.clip(deff, -v_abs, v_abs))
            c = RYG_r(norm(deff))
            ax.barh(y, bar_len, height=0.52, left=CENTER_HALF,
                    color=c, zorder=3, edgecolor="none", alpha=0.95)
            rank  = rk_def.get(r["cat_key"])
            sign  = "+" if deff >= 0 else ""
            r_str = f"#{rank}  " if rank else ""
            ax.text(CENTER_HALF + bar_len + v_lim * 0.04, y,
                    f"{r_str}{sign}{deff:.3f}",
                    ha="left", va="center",
                    color=FG, fontsize=8.5, fontweight="bold", zorder=5)
        else:
            ax.text(CENTER_HALF + (v_lim - CENTER_HALF) * 0.5, y,
                    "n/d", ha="center", va="center", color="#555", fontsize=8)

        # ── Centro: nombre + badge (zona siempre libre de barras) ────────────
        cat_lbl = r["cat_label"].split("\n")[0].strip()
        ax.text(0, y + 0.18, cat_lbl,
                ha="center", va="center",
                color=FG, fontsize=9, fontweight="bold", zorder=5)

        badge_col = COL_EXPLOIT if mm_label == "EXPLOIT" else (COL_RISK if mm_label == "RIESGO" else COL_NEUTRAL)
        badge_bg  = BG_EXPLOIT  if mm_label == "EXPLOIT" else (BG_RISK  if mm_label == "RIESGO" else "#1a2030")
        ax.text(0, y - 0.22, mm_label,
                ha="center", va="center",
                color=badge_col, fontsize=9, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=badge_bg,
                          edgecolor=badge_col, linewidth=1.5, alpha=0.95))

    # Líneas que delimitan la zona central
    for xv in [-CENTER_HALF, CENTER_HALF]:
        ax.axvline(xv, color=GRID, linewidth=0.6, alpha=0.35, zorder=2, linestyle="--")


# ── INPUTS ─────────────────────────────────────────────────────────────────────
team_atk = input("Equipo atacante (ej: SF): ").strip().upper()
team_def = input("Equipo defensor (ej: KC): ").strip().upper()

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
pbp, SEASON = cargar_pbp(SEASON)
pbp["epa"]     = pd.to_numeric(pbp["epa"],     errors="coerce")
pbp["play_id"] = pd.to_numeric(pbp["play_id"], errors="coerce")
print(f"PBP {SEASON}: {len(pbp):,} jugadas REG")

part, _ = cargar_participation(SEASON)
part = part[[
    "nflverse_game_id", "play_id",
    "offense_personnel", "defense_personnel",
    "defense_coverage_type", "defense_man_zone_type",
    "was_pressure",
]]
part = part.rename(columns={"nflverse_game_id": "game_id"})
part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")

merged    = pbp.merge(part, on=["game_id", "play_id"], how="left")
all_plays = merged[merged["play_type"].isin(["pass", "run"]) & merged["epa"].notna()].copy()
print(f"Jugadas totales: {len(all_plays):,}")

all_plays["off_pkg"]    = all_plays["offense_personnel"].apply(parse_off_pkg)
all_plays["def_pkg"]    = all_plays["defense_personnel"].apply(count_dbs).apply(classify_def_pkg)
all_plays["man_zone"]   = all_plays["defense_man_zone_type"].apply(man_zone_label)
all_plays["pressure"]   = all_plays["was_pressure"].apply(pressure_label)
all_plays["red_zone"]   = all_plays["yardline_100"].le(20)
all_plays["third_down"] = all_plays["down"].eq(3)

off_atk = all_plays[all_plays["posteam"] == team_atk].copy()
def_def = all_plays[all_plays["defteam"] == team_def].copy()

if off_atk.empty: raise SystemExit(f"No hay jugadas para {team_atk}.")
if def_def.empty: raise SystemExit(f"No hay jugadas para {team_def}.")
print(f"{team_atk}: {len(off_atk):,} jugadas of.  |  {team_def}: {len(def_def):,} jugadas def.")

# ── RANKINGS ──────────────────────────────────────────────────────────────────
print("Calculando rankings...")
rk_atk_pkg = league_ranks(all_plays, "posteam", "off_pkg",               OFF_PKG_ORDER, ascending=False).get(team_atk, {})
rk_atk_cov = league_ranks(all_plays, "posteam", "defense_coverage_type", COV_ORDER,     ascending=False).get(team_atk, {})
rk_atk_mz  = league_ranks(all_plays, "posteam", "man_zone",              MZ_ORDER,      ascending=False).get(team_atk, {})
rk_atk_pr  = league_ranks(all_plays, "posteam", "pressure",              PRES_ORDER,    ascending=False).get(team_atk, {})

rk_def_pkg = league_ranks(all_plays, "defteam", "off_pkg",               OFF_PKG_ORDER, ascending=True).get(team_def, {})
rk_def_cov = league_ranks(all_plays, "defteam", "defense_coverage_type", COV_ORDER,     ascending=True).get(team_def, {})
rk_def_mz  = league_ranks(all_plays, "defteam", "man_zone",              MZ_ORDER,      ascending=True).get(team_def, {})
rk_def_pr  = league_ranks(all_plays, "defteam", "pressure",              PRES_ORDER,    ascending=True).get(team_def, {})


# ── BUILD ROWS ────────────────────────────────────────────────────────────────
def build_rows(atk_df, def_df, col, cats, cat_labels, lg_epas):
    rows = []
    atk_stats = team_epa_by_cat(atk_df, col, cats)
    def_stats = team_epa_by_cat(def_df, col, cats)
    for cat in cats:
        att_epa, att_n = atk_stats.get(cat, (np.nan, 0))
        def_epa, def_n = def_stats.get(cat, (np.nan, 0))
        lg_epa = lg_epas.get(cat, np.nan)
        if np.isnan(att_epa) and np.isnan(def_epa):
            continue
        rows.append(dict(
            cat_key=cat,
            cat_label=cat_labels.get(cat, cat),
            att_epa=att_epa, att_n=att_n,
            def_epa=def_epa, def_n=def_n,
            lg_epa=lg_epa,
        ))
    return rows


lg_pkg = league_epa_by_cat(all_plays, "off_pkg",               OFF_PKG_ORDER)
lg_cov = league_epa_by_cat(all_plays, "defense_coverage_type", COV_ORDER)
lg_mz  = league_epa_by_cat(all_plays, "man_zone",              MZ_ORDER)
lg_pr  = league_epa_by_cat(all_plays, "pressure",              PRES_ORDER)

rows_pkg = build_rows(off_atk, def_def, "off_pkg",               OFF_PKG_ORDER, OFF_PKG_LABEL, lg_pkg)
rows_cov = build_rows(off_atk, def_def, "defense_coverage_type", COV_ORDER,     COV_LABELS,    lg_cov)
rows_mz  = build_rows(off_atk, def_def, "man_zone",              MZ_ORDER,      {},            lg_mz)
rows_pr  = build_rows(off_atk, def_def, "pressure",              PRES_ORDER,    {},            lg_pr)

def _sit_row(atk_df, def_df, mask_col, key, label, lg_avg):
    a_sub = atk_df[atk_df[mask_col]].dropna(subset=["epa"])
    d_sub = def_df[def_df[mask_col]].dropna(subset=["epa"])
    att_epa = a_sub["epa"].mean() if len(a_sub) >= MIN_SNAPS else np.nan
    def_epa = d_sub["epa"].mean() if len(d_sub) >= MIN_SNAPS else np.nan
    return dict(cat_key=key, cat_label=label,
                att_epa=att_epa, att_n=len(a_sub),
                def_epa=def_epa, def_n=len(d_sub),
                lg_epa=lg_avg)

lg_rz = all_plays[all_plays["red_zone"]].dropna(subset=["epa"])["epa"].mean()
lg_3d = all_plays[all_plays["third_down"]].dropna(subset=["epa"])["epa"].mean()

rows_sit = [
    _sit_row(off_atk, def_def, "red_zone",   "red_zone",   "Red Zone",  lg_rz),
    _sit_row(off_atk, def_def, "third_down", "third_down", "3er Down",  lg_3d),
]

SECTIONS = [
    (rows_pkg, rk_atk_pkg, rk_def_pkg, "PERSONAL OFENSIVO"),
    (rows_cov, rk_atk_cov, rk_def_cov, "COBERTURA DEFENSIVA"),
    (rows_mz,  rk_atk_mz,  rk_def_mz,  "MAN / ZONA"),
    (rows_pr,  rk_atk_pr,  rk_def_pr,  "PRESIÓN"),
    (rows_sit, {},          {},          "SITUACIONES CLAVE"),
]
SECTIONS = [(r, ra, rd, t) for r, ra, rd, t in SECTIONS if r]

N_SEC = len(SECTIONS)
TITLE_H = 0.6
# Altura de cada sección = filas + cabecera
HR_sec = [len(r) + TITLE_H for r, _, _, _ in SECTIONS]
# Fila 0 = header figura
HR = [1.8] + HR_sec

fig = plt.figure(figsize=(15, 0.95 * sum(HR)), facecolor=BG)
gs  = gridspec.GridSpec(
    N_SEC + 1, 1,
    figure=fig,
    height_ratios=HR,
    hspace=0.18,
    left=0.04, right=0.97,
    top=0.955, bottom=0.03,
)

# ── CABECERA ──────────────────────────────────────────────────────────────────
ax_head = fig.add_subplot(gs[0])
ax_head.set_facecolor(CARD)
ax_head.axis("off")

# Separador horizontal entre zona de equipos y zona de leyenda
ax_head.axhline(0.38, color=GRID, linewidth=0.8, alpha=0.5,
                xmin=0.02, xmax=0.98)

# ── Zona superior: logos + nombres + roles ────────────────────────────────────
for team, xlogo, xname, role in [
    (team_atk, 0.08, 0.22, "ATAQUE"),
    (team_def, 0.92, 0.78, "DEFENSA"),
]:
    logo = load_logo(team, base_zoom=0.10)
    if logo:
        ab = AnnotationBbox(logo, (xlogo, 0.70), xycoords="axes fraction",
                            frameon=False, zorder=3)
        ax_head.add_artist(ab)
    ax_head.text(xname, 0.80, team, ha="center", va="center",
                 color=FG, fontsize=22, fontweight="bold",
                 transform=ax_head.transAxes)
    ax_head.text(xname, 0.55, role, ha="center", va="center",
                 color="#888", fontsize=9, transform=ax_head.transAxes)

ax_head.text(0.50, 0.70, "vs", ha="center", va="center",
             color=GRID, fontsize=16, fontweight="bold",
             transform=ax_head.transAxes)

# ── Zona inferior: leyenda badges ────────────────────────────────────────────
for x, badge, desc, col, bg in [
    (0.25, "EXPLOIT", "atacante y defensor fuera de media,\nambos a favor del ataque", COL_EXPLOIT, BG_EXPLOIT),
    (0.50, "NEUTRO",  "situación equilibrada,\nsin ventaja clara", COL_NEUTRAL, "#1a2030"),
    (0.75, "RIESGO",  "defensa superior en esta situación,\nriesgo para el ataque",   COL_RISK,    BG_RISK),
]:
    ax_head.text(x, 0.26, badge,
                 ha="center", va="center", color=col, fontsize=9.5, fontweight="bold",
                 transform=ax_head.transAxes,
                 bbox=dict(boxstyle="round,pad=0.30", facecolor=bg,
                           edgecolor=col, linewidth=1.4))
    ax_head.text(x, 0.10, desc,
                 ha="center", va="center", color="#666", fontsize=6.5,
                 transform=ax_head.transAxes, linespacing=1.4)

for sp in ax_head.spines.values(): sp.set_edgecolor(GRID)

# ── SECCIONES ─────────────────────────────────────────────────────────────────
for s_idx, (rows, rk_a, rk_d, title) in enumerate(SECTIONS):
    ax = fig.add_subplot(gs[s_idx + 1])
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    draw_section(ax, rows, team_atk, team_def, rk_a, rk_d, title)

# ── PIE ───────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.012,
         f"Fuente: nflverse PBP + NGS participation  |  NFL {SEASON}  |  "
         f"#N = ranking en la liga  |  Mín {MIN_SNAPS} snaps  |  "
         f"EXPLOIT: atacante y defensor ambos fuera de media en la misma dirección",
         ha="center", va="bottom", fontsize=6.5, color="#555", fontstyle="italic")

# ── TÍTULO ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.978,
         f"Intel de Matchup  ·  {team_atk} atacando vs {team_def}  ·  NFL {SEASON}",
         ha="center", va="top", fontsize=15, fontweight="bold", color=FG)

# ── WATERMARK ─────────────────────────────────────────────────────────────────
fig.text(0.99, 0.008, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888",
         alpha=0.8, fontstyle="italic")

outfile = f"matchup_intel_{team_atk}_vs_{team_def}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")

# ── RESUMEN TOP 3 ──────────────────────────────────────────────────────────────
all_rows_scored = []
for rows, rk_a, rk_d, section_title in SECTIONS:
    for r in rows:
        _, score = classify_mismatch(r["att_epa"], r["def_epa"], r["lg_epa"])
        all_rows_scored.append({
            **r,
            "score": score,
            "section": section_title,
            "rank_a": rk_a.get(r["cat_key"]),
            "rank_d": rk_d.get(r["cat_key"]),
        })

valid = [r for r in all_rows_scored
         if not np.isnan(r["att_epa"]) and not np.isnan(r["def_epa"])]

top_atk = sorted(valid, key=lambda x: x["score"], reverse=True)[:3]
top_def = sorted(valid, key=lambda x: x["score"])[:3]


def draw_summary_card(ax, row, side, rank_num):
    bg  = BG_EXPLOIT if side == "atk" else BG_RISK
    col = COL_EXPLOIT if side == "atk" else COL_RISK
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_edgecolor(col)
        sp.set_linewidth(2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.05, 0.92, f"#{rank_num}",
            ha="left", va="top", transform=ax.transAxes,
            color=col, fontsize=12, fontweight="bold", alpha=0.65)

    name = row["cat_label"].split("\n")[0].strip()
    ax.text(0.5, 0.74, name,
            ha="center", va="center", transform=ax.transAxes,
            color=FG, fontsize=14, fontweight="bold")

    ax.text(0.5, 0.57, row["section"],
            ha="center", va="center", transform=ax.transAxes,
            color="#666", fontsize=7.5)

    sign_a  = "+" if row["att_epa"] >= 0 else ""
    rk_a_s  = f"  ·  #{row['rank_a']}" if row["rank_a"] else ""
    ax.text(0.5, 0.40,
            f"{team_atk}  EPA  {sign_a}{row['att_epa']:.3f}{rk_a_s}",
            ha="center", va="center", transform=ax.transAxes,
            color=COL_EXPLOIT, fontsize=9.5)

    sign_d  = "+" if row["def_epa"] >= 0 else ""
    rk_d_s  = f"  ·  #{row['rank_d']}" if row["rank_d"] else ""
    ax.text(0.5, 0.25,
            f"{team_def}  EPA cedido  {sign_d}{row['def_epa']:.3f}{rk_d_s}",
            ha="center", va="center", transform=ax.transAxes,
            color=COL_RISK, fontsize=9.5)

    sign_s = "+" if row["score"] >= 0 else ""
    ax.text(0.5, 0.08, f"mismatch  {sign_s}{row['score']:.3f}",
            ha="center", va="center", transform=ax.transAxes,
            color=col, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=BG,
                      edgecolor=col, linewidth=1.2))


fig2 = plt.figure(figsize=(13, 8.5), facecolor=BG)
gs2  = gridspec.GridSpec(
    4, 2,
    figure=fig2,
    height_ratios=[1.1, 2.5, 2.5, 2.5],
    hspace=0.12, wspace=0.07,
    left=0.04, right=0.97,
    top=0.93, bottom=0.05,
)

ax_h = fig2.add_subplot(gs2[0, :])
ax_h.set_facecolor(CARD)
ax_h.axis("off")

for team, xlogo, xname, xtitle, side_col, side_label in [
    (team_atk, 0.08, 0.19, 0.27, COL_EXPLOIT, f"3 ARMAS DE {team_atk}"),
    (team_def, 0.92, 0.81, 0.73, COL_RISK,    f"3 FORTALEZAS DE {team_def}"),
]:
    logo = load_logo(team, base_zoom=0.09)
    if logo:
        ab = AnnotationBbox(logo, (xlogo, 0.55), xycoords="axes fraction",
                            frameon=False, zorder=3)
        ax_h.add_artist(ab)
    ax_h.text(xname, 0.78, team, ha="center", va="center",
              color=FG, fontsize=16, fontweight="bold",
              transform=ax_h.transAxes)
    ax_h.text(xtitle, 0.28, side_label, ha="center", va="center",
              color=side_col, fontsize=10, fontweight="bold",
              transform=ax_h.transAxes)

ax_h.text(0.50, 0.55, "vs", ha="center", va="center",
          color=GRID, fontsize=14, fontweight="bold",
          transform=ax_h.transAxes)
for sp in ax_h.spines.values():
    sp.set_edgecolor(GRID)

for i in range(3):
    ax_l = fig2.add_subplot(gs2[i + 1, 0])
    ax_r = fig2.add_subplot(gs2[i + 1, 1])
    if i < len(top_atk):
        draw_summary_card(ax_l, top_atk[i], "atk", i + 1)
    else:
        ax_l.axis("off")
    if i < len(top_def):
        draw_summary_card(ax_r, top_def[i], "def", i + 1)
    else:
        ax_r.axis("off")

fig2.text(0.5, 0.975,
          f"Resumen Intel  ·  {team_atk} atacando vs {team_def}  ·  NFL {SEASON}",
          ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig2.text(0.5, 0.018,
          f"Fuente: nflverse PBP + NGS  |  NFL {SEASON}  |  "
          f"Mismatch = suma de desviaciones vs media de liga",
          ha="center", va="bottom", fontsize=6.5, color="#555", fontstyle="italic")
fig2.text(0.99, 0.01, "@CuartayDato",
          ha="right", va="bottom", fontsize=9, color="#888888",
          alpha=0.8, fontstyle="italic")

outfile2 = f"matchup_resumen_{team_atk}_vs_{team_def}_{SEASON}.png"
fig2.savefig(outfile2, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig2)
print(f"Guardado: {outfile2}")
