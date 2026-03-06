# match_preview_ranks_dark.py — ATAQUE, DEFENSA y EQUIPOS ESPECIALES (ranking 1=mejor)
# Modo oscuro, PNG listo para redes. Arriba SOLO logos (sin siglas).
# Fuente: nflverse play_by_play_2025 (lectura online).

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Config ===
SEASON       = 2025
URL          = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
EXP_PASS_YDS = 15   # jugada explosiva pase
EXP_RUN_YDS  = 10   # jugada explosiva carrera
FIGSIZE      = (8.7, 13.5)
DPI          = 200
HARD_PENALTY = {"NYJ": 4.5}

# ---------------- Helpers base ----------------
def success_rate(s: pd.Series) -> float:
    e = pd.to_numeric(s, errors="coerce").dropna()
    return float((e > 0).mean()*100) if len(e) else np.nan

def explosive_mask(df_sub: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df_sub["yards_gained"], errors="coerce")
    return (df_sub["play_type"].eq("pass") & (y >= EXP_PASS_YDS)) | \
           (df_sub["play_type"].eq("run")  & (y >= EXP_RUN_YDS))

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# --------- Cálculo ofensivo / defensivo ----------
def compute_offense(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["play_type"].isin(["pass","run"])].copy()
    g = sub.groupby("posteam", dropna=True)
    out = pd.DataFrame(index=g.size().index)
    out["EPA/jugada"]        = g["epa"].mean()
    out["Éxito (%)"]         = g["epa"].apply(success_rate)
    out["EPA/pase"]          = sub.loc[sub["play_type"]=="pass"].groupby("posteam")["epa"].mean()
    out["EPA/carrera"]       = sub.loc[sub["play_type"]=="run"].groupby("posteam")["epa"].mean()
    sub["explosive"]         = explosive_mask(sub)
    out["Explosivas (%)"]    = sub.groupby("posteam")["explosive"].mean().mul(100)
    out["EPA 1º down"]       = sub.loc[sub["down"]==1].groupby("posteam")["epa"].mean()
    out["EPA downs tardíos"] = sub.loc[sub["down"]>=3].groupby("posteam")["epa"].mean()
    out["Distancia media 3º down"] = sub.loc[sub["down"]==3].groupby("posteam")["ydstogo"].mean()
    return out

def compute_defense(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["play_type"].isin(["pass","run"])].copy()
    g = sub.groupby("defteam", dropna=True)
    out = pd.DataFrame(index=g.size().index)
    out["EPA/jugada permitido"]   = g["epa"].mean()
    out["Éxito permitido (%)"]    = g["epa"].apply(success_rate)
    out["EPA/pase permitido"]     = sub.loc[sub["play_type"]=="pass"].groupby("defteam")["epa"].mean()
    out["EPA/carrera permitido"]  = sub.loc[sub["play_type"]=="run"].groupby("defteam")["epa"].mean()
    sub["explosive"]              = explosive_mask(sub)
    out["Explosivas permitidas (%)"] = sub.groupby("defteam")["explosive"].mean().mul(100)
    return out

# --------- Drives: RZ y %RedZone ----------
def compute_drive_level(df: pd.DataFrame):
    dsub = df[df["posteam"].notna()].copy()
    for col in ["game_id","drive","posteam","defteam","yardline_100","touchdown","td_team"]:
        if col not in dsub.columns:
            dsub[col] = np.nan
    to_num(dsub, ["yardline_100","posteam_score_pre","posteam_score_post"])

    groups = dsub.groupby(["game_id","drive"], dropna=False, sort=False)
    recs = []
    for (gid, drv), g in groups:
        posteam = g["posteam"].iloc[0]
        defteam = g["defteam"].iloc[0]
        if pd.isna(posteam):
            continue
        start_y = g["yardline_100"].iloc[0] if "yardline_100" in g.columns else np.nan
        start_y = np.clip(start_y, 1, 99) if pd.notna(start_y) else np.nan
        min_y   = g["yardline_100"].min() if "yardline_100" in g.columns else np.nan
        entered_rz = (pd.notna(min_y) and (min_y <= 20))

        if "td_team" in g.columns:
            td_for = g["td_team"].astype(str).eq(str(posteam)).any()
        else:
            td_for = (g["touchdown"].fillna(False) & g["posteam"].astype(str).eq(str(posteam))).any()

        recs.append({
            "game_id": gid, "drive": drv,
            "posteam": posteam, "defteam": defteam,
            "start_yardline_100": start_y,
            "entered_rz": entered_rz,
            "td_for": td_for
        })
    return pd.DataFrame.from_records(recs)

def compute_redzone_metrics(drv_df: pd.DataFrame):
    rz = drv_df[drv_df["entered_rz"]].copy()

    # ATAQUE: %RedZone = TD / visitas a RZ
    off = rz.groupby("posteam").agg(rz_trips=("entered_rz","size"),
                                    td_drives=("td_for","sum"))
    off["%RedZone"] = (off["td_drives"] / off["rz_trips"]) * 100.0

    # DEFENSA: %RedZone permitido
    deff = rz.groupby("defteam").agg(rz_trips_allowed=("entered_rz","size"),
                                     td_drives_allowed=("td_for","sum"))
    deff["%RedZone permitido"] = (deff["td_drives_allowed"] / deff["rz_trips_allowed"]) * 100.0

    return off[["%RedZone"]], deff[["%RedZone permitido"]]

# --------- Equipos especiales ----------
def compute_special_teams(df: pd.DataFrame) -> pd.DataFrame:
    FG_COL = "field_goal_result" if "field_goal_result" in df.columns else ("fg_result" if "fg_result" in df.columns else None)
    parts = []

    # FG%
    if FG_COL is not None:
        fg = df[(df["play_type"]=="field_goal") & df["posteam"].notna()].copy()
        if not fg.empty:
            fg["made"] = fg[FG_COL].astype(str).str.lower().eq("made")
            parts.append(fg.groupby("posteam")["made"].mean().mul(100).rename("FG%"))

    # EPA/jugada ST
    st_types = {"kickoff","kickoff_return","punt","punt_return","field_goal","extra_point"}
    st_sub = df[df["play_type"].isin(st_types) & df["posteam"].notna()].copy()
    if not st_sub.empty:
        parts.append(st_sub.groupby("posteam")["epa"].mean().rename("EPA/jugada ST"))

    st = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    for c in ["FG%","EPA/jugada ST"]:
        if c not in st.columns: st[c] = np.nan
    return st[["FG%","EPA/jugada ST"]]

# ---------------- Rankings ----------------
def rank_dataframe(df: pd.DataFrame, better_high_cols: list[str], better_low_cols: list[str]) -> pd.DataFrame:
    ranks = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in better_high_cols:
            r = df[col].rank(ascending=False, method="min")
        else:
            r = df[col].rank(ascending=True, method="min")
        ranks[col] = r.astype("Int64")
    return ranks

def rank_to_score(rank_value, max_rank):
    if pd.isna(rank_value) or pd.isna(max_rank) or max_rank < 1:
        return np.nan
    rank_value = float(rank_value); max_rank = float(max_rank)
    return 1.0 if max_rank == 1 else (max_rank - rank_value) / (max_rank - 1)

def safe_max(series):
    try:
        m = series.max()
        return float(m) if pd.notna(m) else np.nan
    except Exception:
        return np.nan

# ---------------- Dibujo ----------------
def load_logo(team: str, base_zoom=0.12):
    path = os.path.join("logos", f"{team}.png")
    if not os.path.exists(path):
        return None, None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        if team in HARD_PENALTY:
            zoom = base_zoom / HARD_PENALTY[team]
        else:
            div = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            zoom = base_zoom / div
        return img, zoom
    except Exception:
        return None, None

def cell(ax, x, y, w, h, color, radius=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={radius}",
                                linewidth=0, facecolor=color, transform=ax.transAxes, zorder=0))

def txt(ax, x, y, s, size=10, weight=None, ha="left", va="center", color="#EDEDED"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, fontweight=weight, color=color)

def fmt_val(v, is_pct=False, nd=2):
    if pd.isna(v): return "–"
    return f"{v:.1f}%" if is_pct else f"{v:.{nd}f}"

def draw_png(team_a, team_b, off, deff, st, off_r, deff_r, st_r, out_path):
    off_rows  = ["EPA/jugada","Éxito (%)","EPA/pase","EPA/carrera","Explosivas (%)",
                 "EPA 1º down","EPA downs tardíos","Distancia media 3º down","%RedZone"]
    deff_rows = ["EPA/jugada permitido","Éxito permitido (%)","EPA/pase permitido","EPA/carrera permitido",
                 "Explosivas permitidas (%)","%RedZone permitido"]
    st_rows   = ["FG%","EPA/jugada ST"]

    cmap    = LinearSegmentedColormap.from_list("r2g", ["#ff6b6b","#ffd166","#06d6a0"])
    NEUTRAL = "#4A4A4A"; BG = "#0f1115"; CARD = "#151924"; INK = "#F2F3F5"; SUBINK="#C9CDD6"; ACCENT="#2d6cdf"

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG); ax.axis("off")
    cell(ax, 0.04, 0.06, 0.92, 0.90, CARD, radius=0.03)

    # Posiciones de columnas
    x_metric = 0.10
    xA, xB = 0.50, 0.78

    # --- SOLO LOGOS arriba (sin siglas ni "píldoras") ---
    logo_y = 0.927
    for x_center, abbr in [(xA, team_a), (xB, team_b)]:
        img, z = load_logo(abbr, 0.10)
        if img is not None:
            ab = AnnotationBbox(OffsetImage(img, zoom=z, resample=True), (x_center, logo_y),
                                frameon=False, xycoords=ax.transAxes)
            ax.add_artist(ab)

    # Layout de filas
    n_off, n_def, n_st = len(off_rows), len(deff_rows), len(st_rows)
    n_rows = n_off + n_def + n_st
    top_y, bottom_y = 0.885, 0.115
    usable_y = top_y - bottom_y
    section_gap_factor = 1.10
    row_gap_factor     = 1.15
    n_gaps = 2
    step  = usable_y / (n_rows * row_gap_factor + n_gaps * section_gap_factor)
    box_w = 0.19
    box_h = step * 0.70
    y = top_y

    off_max = {c: safe_max(off_r[c])  for c in off_rows  if c in off_r.columns}
    def_max = {c: safe_max(deff_r[c]) for c in deff_rows if c in deff_r.columns}
    st_max  = {c: safe_max(st_r[c])   for c in st_rows   if c in st_r.columns}

    def divider(title):
        nonlocal y
        ax.plot([0.07, 0.93], [y - step*0.25, y - step*0.25], color=ACCENT, lw=1.2, alpha=0.4, transform=ax.transAxes)
        txt(ax, 0.07, y - step*0.10, title, 13, "bold", color=INK)
        y -= step * section_gap_factor

    def draw_row(label, vA, rA, vB, rB, max_rank, is_pct=False):
        nonlocal y
        txt(ax, x_metric, y - box_h*0.05, label, 10.5, color=INK)

        sA = rank_to_score(rA, max_rank); sB = rank_to_score(rB, max_rank)
        colA = cmap(float(sA)) if pd.notna(sA) else NEUTRAL
        colB = cmap(float(sB)) if pd.notna(sB) else NEUTRAL

        cell(ax, xA - box_w/2, y - box_h/2, box_w, box_h, colA, 0.012)
        txt(ax, xA - box_w/2 + 0.012, y, fmt_val(vA, is_pct), 10.5, color="#0f1115")
        txt(ax, xA + box_w/2 - 0.012, y, "–" if pd.isna(rA) else f"#{int(rA)}", 9.5, ha="right", color="#0f1115")

        cell(ax, xB - box_w/2, y - box_h/2, box_w, box_h, colB, 0.012)
        txt(ax, xB - box_w/2 + 0.012, y, fmt_val(vB, is_pct), 10.5, color="#0f1115")
        txt(ax, xB + box_w/2 - 0.012, y, "–" if pd.isna(rB) else f"#{int(rB)}", 9.5, ha="right", color="#0f1115")

        y -= step * row_gap_factor

    # ----- Secciones -----
    divider("ATAQUE")
    for r in off_rows:
        vA = off.loc[team_a, r] if team_a in off.index else np.nan
        vB = off.loc[team_b, r] if team_b in off.index else np.nan
        rA = off_r.loc[team_a, r] if (team_a in off_r.index and r in off_r.columns) else np.nan
        rB = off_r.loc[team_b, r] if (team_b in off_r.index and r in off_r.columns) else np.nan
        draw_row(r, vA, rA, vB, rB, max_rank=off_max.get(r, np.nan),
                 is_pct=(r in ["Éxito (%)","Explosivas (%)","%RedZone"]))

    divider("DEFENSA")
    for r in deff_rows:
        vA = deff.loc[team_a, r] if team_a in deff.index else np.nan
        vB = deff.loc[team_b, r] if team_b in deff.index else np.nan
        rA = deff_r.loc[team_a, r] if (team_a in deff_r.index and r in deff_r.columns) else np.nan
        rB = deff_r.loc[team_b, r] if (team_b in deff_r.index and r in deff_r.columns) else np.nan
        draw_row(r, vA, rA, vB, rB, max_rank=def_max.get(r, np.nan),
                 is_pct=("Éxito" in r or "Explosivas" in r or "RedZone" in r))

    divider("EQUIPOS ESPECIALES")
    for r in st_rows:
        vA = st.loc[team_a, r] if (team_a in st.index and r in st.columns) else np.nan
        vB = st.loc[team_b, r] if (team_b in st.index and r in st.columns) else np.nan
        rA = st_r.loc[team_a, r] if (team_a in st_r.index and r in st_r.columns) else np.nan
        rB = st_r.loc[team_b, r] if (team_b in st_r.index and r in st_r.columns) else np.nan
        draw_row(r, vA, rA, vB, rB, max_rank=st_max.get(r, np.nan),
                 is_pct=(r=="FG%"))

    # --- Firma sutil con @CuartayDato en la esquina inferior derecha ---
    ax.text(0.90, 0.065, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

# ---------------- Main ----------------
if __name__ == "__main__":
    team_a = input("Equipo A (siglas exactas, p.ej. SF): ").strip().upper()
    team_b = input("Equipo B (siglas exactas, p.ej. DAL): ").strip().upper()

    print(f"\nCargando play-by-play {SEASON}...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    to_num(df, ["epa","yards_gained","ydstogo","down","return_yards","yardline_100",
                "posteam_score_pre","posteam_score_post"])

    off_basic  = compute_offense(df)
    deff_basic = compute_defense(df)

    drv_df = compute_drive_level(df)
    off_rz, deff_rz = compute_redzone_metrics(drv_df)

    off  = off_basic.join(off_rz,  how="left")
    deff = deff_basic.join(deff_rz, how="left")

    st = compute_special_teams(df)

    # Rankings
    off_high = ["EPA/jugada","Éxito (%)","EPA/pase","EPA/carrera","Explosivas (%)",
                "EPA 1º down","EPA downs tardíos","%RedZone"]
    off_ranks = rank_dataframe(off, better_high_cols=off_high, better_low_cols=["Distancia media 3º down"])

    deff_ranks = rank_dataframe(deff, better_high_cols=[], better_low_cols=list(deff.columns))

    st_high = ["FG%","EPA/jugada ST"]
    st_low  = []
    st_ranks = rank_dataframe(st, better_high_cols=st_high, better_low_cols=st_low)

    out_file = f"preview_{team_a}_vs_{team_b}_{SEASON}.png"
    draw_png(team_a, team_b, off, deff, st, off_ranks, deff_ranks, st_ranks, out_file)
    print(f"\nPNG generado: {out_file}")
    print("Asegurate de tener los PNG en ./logos/SIGLA.png (ej.: logos/SF.png).")
